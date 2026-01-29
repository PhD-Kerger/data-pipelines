import os
import glob
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path

import pyarrow_ops as pa_ops

from utils import DataPipelineLogger


class Tier:

    def __init__(
        self,
        input_data_dir_path,
        meta_data_dir_path,
        export_data_dir_path,
        logs_data_dir_path,
        zone_id_mappings,
        processor_class,
        zone_ids=None,
        osrm_enabled=False,
        osrm_alternative_percentage=0,
        processing_steps=["trips", "availability", "demand"],
    ):
        if not hasattr(self, "initialized"):
            self.initialized = True
            self.input_data_dir_path = input_data_dir_path
            self.meta_data_dir_path = meta_data_dir_path
            self.export_data_dir_path = export_data_dir_path
            self.log_file = Path(logs_data_dir_path) / "logs.log"
            self.zone_ids = zone_ids
            self.zone_id_mappings = zone_id_mappings
            self.OSRM_ENABLED = osrm_enabled
            self.OSRM_ALTERNATIVE_PERCENTAGE = osrm_alternative_percentage
            self.db_connection = None
            self.processing_steps = processing_steps
            self.processor_class = processor_class

            # Cache for geospatial data
            self._rheinland_pfalz_geometry = None

            # Setup logger
            self.logger = DataPipelineLogger.get_logger(
                name=self.__class__.__name__, log_file_path=self.log_file
            )

            self._initialize_database()

    def _create_zone_id_mapping_table(self):
        """Create a DuckDB table for zone ID mappings"""
        try:
            # Create a mapping table for zone IDs
            mapping_data = {
                "zone_id": list(self.zone_id_mappings.keys()),
                "network_name": list(self.zone_id_mappings.values()),
            }
            mapping_table = pa.table(mapping_data)

            # Register the table with DuckDB and create the table
            self.db_connection.register("temp_mapping_table", mapping_table)
            self.db_connection.execute(
                "CREATE OR REPLACE TABLE zone_id_mappings AS SELECT * FROM temp_mapping_table"
            )
            self.db_connection.unregister("temp_mapping_table")

            self.logger.info("Zone ID mapping table created successfully")
        except Exception as e:
            self.logger.error(f"Error creating zone ID mapping table: {e}")

    def _load_rheinland_pfalz_geometry(self):
        """Load the Rheinland-Pfalz shapefile for geospatial checks"""
        if self._rheinland_pfalz_geometry is not None:
            return self._rheinland_pfalz_geometry

        try:
            # Load the Rheinland-Pfalz shapefile
            geojson_path = Path(
                "./data/internal/geodata/rheinland_pfalz/rheinland_pfalz.shp"
            )
            if not geojson_path.exists():
                self.logger.error(f"Shapefile {geojson_path} does not exist.")
                return None

            gdf = gpd.read_file(geojson_path)
            # Get the unified geometry (union of all polygons in case there are multiple)
            self._rheinland_pfalz_geometry = gdf.unary_union
            self.logger.info("Successfully loaded Rheinland-Pfalz geometry")
            return self._rheinland_pfalz_geometry

        except Exception as e:
            self.logger.error(f"Error loading Rheinland-Pfalz geometry: {e}")
            return None

    def _is_point_in_rheinland_pfalz(self, lat, lng):
        """Check if a point (lat, lng) is within Rheinland-Pfalz boundaries"""
        try:
            geometry = self._load_rheinland_pfalz_geometry()
            if geometry is None:
                return None

            point = Point(lng, lat)  # Note: shapely uses (lng, lat) order
            return geometry.contains(point)

        except Exception as e:
            self.logger.warning(
                f"Error checking point ({lat}, {lng}) in Rheinland-Pfalz: {e}"
            )
            return None

    def _adjust_zone_id_based_on_geography(self, zone_id, lat, lng):
        """Adjust ZoneID based on geographical location for MANNHEIM-LUDWIGSHAFEN"""
        if zone_id != "MANNHEIM-LUDWIGSHAFEN":
            return zone_id

        # Check if the point is in Rheinland-Pfalz
        in_rheinland_pfalz = self._is_point_in_rheinland_pfalz(lat, lng)

        if in_rheinland_pfalz is True:
            # Point is in Rheinland-Pfalz, so it's Ludwigshafen
            return "LUDWIGSHAFEN"
        elif in_rheinland_pfalz is False:
            # Point is not in Rheinland-Pfalz, so it's Mannheim (Baden-Württemberg)
            return "MANNHEIM"
        else:
            # Could not determine, keep original
            self.logger.warning(
                f"Could not determine location for point ({lat}, {lng}), keeping original zone: {zone_id}"
            )
            return zone_id

    def _apply_zone_id_geographic_adjustment(self):
        """Apply geographic zone ID adjustments by creating a view with adjusted ZoneIDs"""
        try:
            self.logger.info("Starting geographic zone ID adjustment...")

            # First, check if there are any MANNHEIM-LUDWIGSHAFEN records to adjust
            check_query = """
            SELECT COUNT(*) 
            FROM vehicles 
            WHERE ZoneID = 'MANNHEIM-LUDWIGSHAFEN'
            """
            count_result = self.db_connection.execute(check_query).fetchone()
            records_to_adjust = count_result[0] if count_result else 0

            if records_to_adjust == 0:
                self.logger.info("No MANNHEIM-LUDWIGSHAFEN records found to adjust")
                return

            self.logger.info(
                f"Found {records_to_adjust} MANNHEIM-LUDWIGSHAFEN records to adjust"
            )

            # Get all unique coordinates for MANNHEIM-LUDWIGSHAFEN records
            coord_query = """
            SELECT DISTINCT lat_rounded, lng_rounded, ZoneID
            FROM vehicles 
            WHERE ZoneID = 'MANNHEIM-LUDWIGSHAFEN'
            AND lat_rounded IS NOT NULL AND lng_rounded IS NOT NULL
            """
            coord_result = self.db_connection.execute(coord_query).fetchall()

            if not coord_result:
                self.logger.warning(
                    "No valid coordinates found for MANNHEIM-LUDWIGSHAFEN records"
                )
                return

            # Build mapping of coordinates to adjusted zone IDs
            coord_adjustments = {}
            adjusted_count = 0

            for lat, lng, zone_id in coord_result:
                adjusted_zone = self._adjust_zone_id_based_on_geography(
                    zone_id, lat, lng
                )
                if adjusted_zone != zone_id:
                    coord_adjustments[(lat, lng)] = adjusted_zone
                    adjusted_count += 1

            if not coord_adjustments:
                self.logger.info("No zone adjustments needed based on geography")
                return

            self.logger.info(
                f"Will adjust {len(coord_adjustments)} unique coordinate locations"
            )

            # Create a new view with adjusted ZoneIDs using CASE statements
            case_conditions = []
            for (lat, lng), new_zone_id in coord_adjustments.items():
                case_conditions.append(
                    f"WHEN (lat_rounded = {lat} AND lng_rounded = {lng}) THEN '{new_zone_id}'"
                )

            case_statement = " ".join(case_conditions)

            # Recreate the vehicles view with geographic adjustments
            # First, get the current view definition (either filtered or unfiltered)
            current_view_base = "vehicles_raw"

            # Add filter conditions if they exist
            filter_conditions = self._build_filter_conditions(self.zone_ids)
            where_clause = f"WHERE 1=1 {filter_conditions}" if filter_conditions else ""

            adjusted_view_query = f"""
            CREATE OR REPLACE VIEW vehicles AS 
            SELECT 
                LicensePlate,
                Timestamp,
                Lat,
                Lng,
                lat_rounded,
                lng_rounded,
                CASE 
                    {case_statement}
                    ELSE ZoneID 
                END AS ZoneID,
                BatteryLevel,
                CurrentRangeMeters,
                MaxSpeed,
                State,
                IsRentable
            FROM {current_view_base}
            {where_clause}
            """

            # Execute the query to replace the view
            self.db_connection.execute(adjusted_view_query)

            # Verify the updates
            verification_query = """
            SELECT COUNT(*) 
            FROM vehicles 
            WHERE ZoneID = 'MANNHEIM-LUDWIGSHAFEN'
            """
            remaining_result = self.db_connection.execute(verification_query).fetchone()
            remaining_records = remaining_result[0] if remaining_result else 0

            adjusted_records = records_to_adjust - remaining_records
            self.logger.info(
                f"Geographic zone ID adjustment completed: {adjusted_records} records updated"
            )

            if remaining_records > 0:
                self.logger.warning(
                    f"{remaining_records} MANNHEIM-LUDWIGSHAFEN records remain (could not determine geography)"
                )

        except Exception as e:
            self.logger.error(f"Error applying geographic zone ID adjustments: {e}")

    def _initialize_database(self):
        """Initialize DuckDB database with views pointing to Parquet files"""
        # Set the data directory path
        data_dir = Path(__file__).parent.parent / self.input_data_dir_path
        if not data_dir.exists():
            self.logger.error(f"Data directory {data_dir} does not exist.")
            return

            # Find all date directories in yyyy-mm-dd format
        date_pattern = os.path.join(data_dir, "????-??-??")
        date_dirs = glob.glob(date_pattern)

        if not date_dirs:
            self.logger.warning(f"No date directories found in {data_dir}")
            return

        # Collect paths for tier vehicle status files
        vehicle_paths = []

        for date_dir in sorted(date_dirs):
            vehicles_file = os.path.join(date_dir, f"{os.path.basename(date_dir)}.parquet")

            if os.path.exists(vehicles_file):
                vehicle_paths.append(vehicles_file)

        # Connect to DuckDB
        con = duckdb.connect(database=f"./data/duckdb/tier.duckdb", read_only=False)
        # Store the connection for later use
        self.db_connection = con

        # Create tables/views
        self.logger.info("Creating tables in DuckDB...")

        # Create base view that points to the Parquet files
        if vehicle_paths:
            con.execute(
                f"CREATE OR REPLACE TABLE vehicles_raw AS SELECT *, ROUND(Lat, 3) AS lat_rounded, ROUND(Lng, 3) AS lng_rounded FROM read_parquet([{', '.join([f'\'{path}\'' for path in vehicle_paths])}])"
            )
            self.logger.info(
                f"Created vehicles_raw table with {len(vehicle_paths)} files"
            )
            # print number of rows in vehicles_raw
            row_count = con.execute("SELECT COUNT(*) FROM vehicles_raw").fetchone()[0]
            self.logger.info(f"vehicles_raw contains {row_count} rows")
        else:
            self.logger.error("No vehicle status files found to create vehicles_raw")
            return

        # Build filter conditions
        filter_conditions = self._build_filter_conditions(self.zone_ids)

        # Create filtered views based on initialization parameters
        if filter_conditions:
            self.logger.info(f"Applying filters: {filter_conditions}")

            # Create filtered view
            con.execute(
                f"""
                CREATE OR REPLACE VIEW vehicles AS 
                SELECT * FROM vehicles_raw
                WHERE 1=1 {filter_conditions}
            """
            )
            # get number of rows in vehicles view
            row_count = con.execute("SELECT COUNT(*) FROM vehicles").fetchone()[0]
            self.logger.info(f"filtered vehicles view contains {row_count} rows")

            self.logger.info("Created filtered view based on zone/vehicle type filters")
        else:
            # No filters - create simple alias
            con.execute("CREATE OR REPLACE VIEW vehicles AS SELECT * FROM vehicles_raw")
            self.logger.info(
                "Created unfiltered view (no zone/vehicle type filters applied)"
            )

        # Adjust ZoneID with _adjust_zone_id_based_on_geography
        self._apply_zone_id_geographic_adjustment()

        # Create zone ID mapping table
        self._create_zone_id_mapping_table()

        self.logger.info("DuckDB initialization completed successfully!")

        self._get_unique_location_coordinates()
        self.logger.info("Unique location coordinates extracted and saved")

    def _get_unique_location_coordinates(self):
        """Extract unique coordinates and append to location_coordinates.parquet"""
        try:
            # Define the meta data file path first
            meta_dir = Path(self.meta_data_dir_path)
            meta_dir.mkdir(parents=True, exist_ok=True)
            coordinates_file = meta_dir / "location_coordinates.parquet"

            # Get unique coordinates from vehicles table
            query = """
            SELECT DISTINCT lat_rounded AS lat, lng_rounded AS lng
            FROM vehicles 
            WHERE lat_rounded IS NOT NULL AND lng_rounded IS NOT NULL 
            ORDER BY lat_rounded, lng_rounded
            """
            result = self.db_connection.execute(query).fetchall()

            if not result:
                self.logger.warning("No coordinates found in vehicles table")
                return

            # Extract coordinates from query result
            new_coords = [(row[0], row[1]) for row in result]

            # Handle existing file
            if coordinates_file.exists():
                # Read existing data
                existing_data = pq.read_table(coordinates_file)
                existing_coords = set(
                    zip(
                        existing_data.column("lat").to_pylist(),
                        existing_data.column("lng").to_pylist(),
                    )
                )
                max_location_id = max(existing_data.column("location_id").to_pylist())

                # Filter out coordinates that already exist
                new_coords = [
                    (lat, lng)
                    for lat, lng in new_coords
                    if (lat, lng) not in existing_coords
                ]

                if not new_coords:
                    self.logger.info("No new coordinates to add")
                    # Still create the DuckDB table from existing file
                    self.db_connection.execute(
                        f"""
                        CREATE OR REPLACE TABLE location_coordinates AS
                        SELECT * FROM parquet_scan('{coordinates_file}')
                        """
                    )
                    return

                self.logger.info(
                    f"Adding {len(new_coords)} new coordinates to existing file"
                )
            else:
                max_location_id = 0
                self.logger.info(
                    f"Creating new coordinates file with {len(new_coords)} coordinates"
                )

            # Create data for new coordinates only
            if new_coords:
                new_lats, new_lngs = zip(*new_coords)
                new_location_ids = [
                    max_location_id + i + 1 for i in range(len(new_coords))
                ]

                new_data = pa.table(
                    {"location_id": new_location_ids, "lat": new_lats, "lng": new_lngs}
                )

                # Append to existing file or create new one
                if coordinates_file.exists():
                    existing_data = pq.read_table(coordinates_file)
                    combined_data = pa.concat_tables([existing_data, new_data])
                    pq.write_table(combined_data, coordinates_file, compression="BROTLI")
                else:
                    pq.write_table(new_data, coordinates_file, compression="BROTLI")

                self.logger.info(f"Saved coordinates to {coordinates_file}")

            # Create DuckDB table from the parquet file
            self.db_connection.execute(
                f"""
                CREATE OR REPLACE TABLE location_coordinates AS
                SELECT * FROM parquet_scan('{coordinates_file}')
                """
            )

        except Exception as e:
            self.logger.error(f"Error extracting coordinates: {e}")

    def _build_filter_conditions(self, zone_ids=None):
        """Build SQL filter conditions for zone IDs"""
        conditions = []

        if zone_ids and len(zone_ids) > 0:
            zone_list = ", ".join([f"'{zone_id}'" for zone_id in zone_ids])
            conditions.append(f"ZoneID IN ({zone_list})")

        return " AND " + " AND ".join(conditions) if conditions else ""

    def trip_generator(self):
        """Generate trips from the vehicles table and save to daily parquet files"""
        try:
            self.logger.info("Starting trip generation...")
            
            query = f"""
            WITH filtered_vehicles AS (
                SELECT
                    LicensePlate AS vehicle_id,
                    CAST(to_timestamp(Timestamp) AS TIMESTAMPTZ) AS Timestamp,
                    v.lat_rounded AS Lat,
                    v.lng_rounded AS Lng,
                    ZoneID,
                    CAST(lc.location_id AS INTEGER) AS location_id,
                    zm.network_name,
                    CurrentRangeMeters,
                    BatteryLevel,
                    MaxSpeed
                FROM vehicles v
                LEFT JOIN location_coordinates lc ON (v.lat_rounded = lc.lat AND v.lng_rounded = lc.lng)
                LEFT JOIN zone_id_mappings zm ON v.ZoneID = zm.zone_id
            ),
            trips as (
                SELECT
                    *,
                    vehicle_id,
                    Timestamp AS timestamp_return,
                    NULL::VARCHAR AS station_id_lend,
                    NULL::VARCHAR AS station_id_return,
                    NULL::INTEGER AS station_name_id_lend,
                    NULL::INTEGER AS station_name_id_return,
                    network_name AS network_name_return,
                    location_id AS location_id_return,
                    BatteryLevel::SMALLINT AS pedelec_battery_return,
                    CurrentRangeMeters::INTEGER AS current_range_meters_return
                FROM (
                    SELECT
                        *,
                        LAG(Timestamp) OVER (PARTITION BY vehicle_id ORDER BY Timestamp) AS timestamp_lend,
                        LAG(location_id) OVER (PARTITION BY vehicle_id ORDER BY Timestamp) AS prev_location_id,
                        LAG(Timestamp) OVER (PARTITION BY vehicle_id ORDER BY Timestamp) AS prev_timestamp,
                        LAG(network_name) OVER (PARTITION BY vehicle_id ORDER BY Timestamp) AS network_name_lend,
                        LAG(location_id) OVER (PARTITION BY vehicle_id ORDER BY Timestamp) AS location_id_lend,
                        LAG(CurrentRangeMeters) OVER (PARTITION BY vehicle_id ORDER BY Timestamp)::INTEGER AS current_range_meters_lend,
                        LAG(BatteryLevel) OVER (PARTITION BY vehicle_id ORDER BY Timestamp)::SMALLINT AS pedelec_battery_lend
                    FROM filtered_vehicles
                )
                WHERE
                    prev_location_id IS NOT NULL
                    AND prev_location_id != location_id
                    AND (pedelec_battery_lend IS NULL OR pedelec_battery_return IS NULL OR pedelec_battery_lend != pedelec_battery_return)
            )
            SELECT DISTINCT
                vehicle_id,
                timestamp_lend,
                timestamp_return,
                station_id_lend,
                station_id_return,
                station_name_id_lend,
                station_name_id_return,
                network_name_lend,
                network_name_return,
                location_id_lend,
                location_id_return,
                pedelec_battery_lend,
                pedelec_battery_return,
                current_range_meters_lend,
                current_range_meters_return,
                DATE(timestamp_lend) AS trip_date
            FROM trips
            ORDER BY vehicle_id, timestamp_lend;
            """
            result_sql = self.db_connection.execute(query)
            
            result = result_sql.fetch_arrow_table()

            if result.num_rows == 0:
                self.logger.warning("No trips generated")
                return

            self.logger.info(f"Generated {result.num_rows} trips")

            # Group by trip_date
            trip_date_col = result.column("trip_date")
            unique_dates = pa.compute.unique(trip_date_col).to_pylist()

            # Save to files directly
            export_dir = Path(self.export_data_dir_path) / "trips"
            export_dir.mkdir(parents=True, exist_ok=True)

            for date_key in unique_dates:
                # Filter rows for this date
                mask = pa.compute.equal(trip_date_col, date_key)
                daily_table = result.filter(mask)

                if daily_table.num_rows == 0:
                    continue

                # Drop the trip_date column and add predicted column
                column_names = daily_table.schema.names
                columns_to_keep = [col for col in column_names if col != "trip_date"]
                daily_table = daily_table.select(columns_to_keep)

                # Add predicted column
                predicted_col = pa.array([False] * daily_table.num_rows, type=pa.bool_())
                daily_table = daily_table.append_column("predicted", predicted_col)

                # Save to parquet file named by date
                filename = f"{date_key}.parquet"
                file_path = export_dir / filename

                if file_path.exists():
                    # Read existing data and append
                    existing_data = pq.read_table(file_path)
                    combined_data = pa.concat_tables([existing_data, daily_table])
                    # sort by vehicle_id and timestamp_lend
                    combined_data = combined_data.sort_by(
                        [("vehicle_id", "ascending"), ("timestamp_lend", "ascending")]
                    )
                    key_columns = ["vehicle_id", "timestamp_lend"]
                    # make a distinction to avoid duplicate rows
                    deduped_table = pa_ops.drop_duplicates(
                        combined_data, key_columns, keep="first"
                    )
                    pq.write_table(deduped_table, file_path, compression="BROTLI")
                    self.logger.info(
                        f"Added {daily_table.num_rows} trip records to existing file {filename}"
                    )
                else:
                    # Create new file
                    pq.write_table(daily_table, file_path, compression="BROTLI")
                    self.logger.info(
                        f"Created {filename} with {daily_table.num_rows} trip records"
                    )

            self.logger.info(f"Trip generation completed. Files saved to {export_dir}")

        except Exception as e:
            self.logger.error(f"Error generating trips: {e}")

        except Exception as e:
            self.logger.error(f"Error generating trips: {e}")

    def demand_generator(self):
        """Generate rent and return demand from trip files and save to daily parquet files"""
        try:
            self.logger.info("Starting demand generation...")

            # Check if trips directory exists
            trips_dir = Path(self.export_data_dir_path) / "trips"
            if not trips_dir.exists():
                self.logger.error(
                    f"Trips directory {trips_dir} does not exist. Run trip_generator first."
                )
                return

            # Get all trip files
            trip_files = list(trips_dir.glob("*.parquet"))
            if not trip_files:
                self.logger.warning(f"No trip files found in {trips_dir}")
                return

            self.logger.info(f"Found {len(trip_files)} trip files to process")

            # Process all trip files at once
            trip_files_str = ", ".join([f"'{file}'" for file in trip_files])

            query = f"""
            WITH all_trips AS (
                SELECT * FROM read_parquet([{trip_files_str}])
            ),
            rent_demand AS (
                SELECT 
                    location_id_lend AS location_id,
                    timestamp_lend AS timestamp,
                    network_name_lend AS network_name,
                    NULL::INTEGER AS station_name_id,
                    NULL::VARCHAR AS station_id,
                    COUNT(*)::SMALLINT AS n_lends,
                    0::SMALLINT AS n_returns
                FROM all_trips
                WHERE location_id_lend IS NOT NULL AND timestamp_lend IS NOT NULL
                GROUP BY location_id_lend, timestamp_lend, network_name_lend
            ),
            return_demand AS (
                SELECT 
                    location_id_return AS location_id,
                    timestamp_return AS timestamp,
                    network_name_return AS network_name,
                    NULL::INTEGER AS station_name_id,
                    NULL::VARCHAR AS station_id,
                    0::SMALLINT AS n_lends,
                    COUNT(*)::SMALLINT AS n_returns
                FROM all_trips
                WHERE location_id_return IS NOT NULL AND timestamp_return IS NOT NULL
                GROUP BY location_id_return, timestamp_return, network_name_return
            ),
            combined_demand AS (
                SELECT * FROM rent_demand
                UNION ALL
                SELECT * FROM return_demand
            )
            SELECT
                location_id,
                timestamp,
                network_name,
                station_name_id,
                station_id,
                SUM(n_lends)::SMALLINT AS n_lends,
                SUM(n_returns)::SMALLINT AS n_returns,
                DATE(timestamp) AS demand_date
            FROM combined_demand
            GROUP BY location_id, timestamp, network_name, station_name_id, station_id
            ORDER BY location_id, timestamp;
            """
            self.logger.info("Executing demand generation query...")
            result_sql = self.db_connection.execute(query)
            result = result_sql.fetch_arrow_table()

            if result.num_rows == 0:
                self.logger.warning("No demand data generated")
                return

            self.logger.info(f"Generated {result.num_rows} demand records")

            # Group by demand_date
            demand_date_col = result.column("demand_date")
            unique_dates = pa.compute.unique(demand_date_col).to_pylist()

            # Save to files directly
            export_dir = Path(self.export_data_dir_path) / "demand"
            export_dir.mkdir(parents=True, exist_ok=True)

            for date_key in unique_dates:
                # Filter rows for this date
                mask = pa.compute.equal(demand_date_col, date_key)
                daily_table = result.filter(mask)

                if daily_table.num_rows == 0:
                    continue

                # Drop the demand_date column
                column_names = daily_table.schema.names
                columns_to_keep = [col for col in column_names if col != "demand_date"]
                daily_table = daily_table.select(columns_to_keep)

                # Save to parquet file named by date
                filename = f"{date_key}.parquet"
                file_path = export_dir / filename

                if file_path.exists():
                    # Read existing data and append
                    existing_data = pq.read_table(file_path)
                    combined_data = pa.concat_tables([existing_data, daily_table])
                    # sort by vehicle_id and timestamp_lend
                    combined_data = combined_data.sort_by(
                        [("vehicle_id", "ascending"), ("timestamp_lend", "ascending")]
                    )
                    key_columns = ["vehicle_id", "timestamp_lend"]
                    # make a distinction to avoid duplicate rows
                    deduped_table = pa_ops.drop_duplicates(
                        combined_data, key_columns, keep="first"
                    )
                    pq.write_table(deduped_table, file_path, compression="BROTLI")
                    self.logger.info(
                        f"Added {daily_table.num_rows} demand records to existing file {filename}"
                    )
                else:
                    # Create new file
                    pq.write_table(daily_table, file_path, compression="BROTLI")
                    self.logger.info(
                        f"Created {filename} with {daily_table.num_rows} demand records"
                    )

            self.logger.info(
                f"Demand generation completed. Files saved to {export_dir}"
            )

        except Exception as e:
            self.logger.error(f"Error generating demand: {e}")

    def availability_generator(self):
        """Generate vehicle availability data grouped by location and timestamp from raw data"""
        try:
            self.logger.info("Starting availability generation...")

            query = """
            WITH filtered_vehicles AS (
                SELECT
                    v.Timestamp,
                    v.Lat,
                    v.Lng,
                    v.ZoneID
                FROM vehicles v
            ),
            availability_data AS (
                SELECT
                    CAST(lc.location_id AS INTEGER) AS location_id,
                    CAST(to_timestamp(fv.Timestamp) AS TIMESTAMPTZ) AS timestamp,
                    fv.ZoneID,
                    zm.network_name,
                    NULL::INTEGER AS station_name_id,
                    NULL::VARCHAR AS station_id,
                    COUNT(*)::SMALLINT AS n_vehicles,
                    DATE(CAST(to_timestamp(fv.Timestamp) AS TIMESTAMPTZ)) AS availability_date
                FROM filtered_vehicles fv
                LEFT JOIN location_coordinates lc ON (fv.Lat = lc.lat AND fv.Lng = lc.lng)
                LEFT JOIN zone_id_mappings zm ON fv.ZoneID = zm.zone_id
                WHERE lc.location_id IS NOT NULL
                GROUP BY lc.location_id, fv.Timestamp, fv.ZoneID, zm.network_name
                ORDER BY timestamp, location_id
            )
            SELECT 
                location_id,
                timestamp,
                network_name,
                station_name_id,
                station_id,
                n_vehicles,
                availability_date
            FROM availability_data;
            """

            self.logger.info("Executing availability generation query...")
            result_sql = self.db_connection.execute(query)
            result = result_sql.fetch_arrow_table()

            if result.num_rows == 0:
                self.logger.warning("No availability data generated")
                return

            self.logger.info(f"Generated {result.num_rows} availability records")

            # Group by availability_date
            availability_date_col = result.column("availability_date")
            unique_dates = pa.compute.unique(availability_date_col).to_pylist()

            # Save to files directly
            export_dir = Path(self.export_data_dir_path) / "availability"
            export_dir.mkdir(parents=True, exist_ok=True)

            for date_key in unique_dates:
                # Filter rows for this date
                mask = pa.compute.equal(availability_date_col, date_key)
                daily_table = result.filter(mask)

                if daily_table.num_rows == 0:
                    continue

                # Drop the availability_date column
                column_names = daily_table.schema.names
                columns_to_keep = [col for col in column_names if col != "availability_date"]
                daily_table = daily_table.select(columns_to_keep)

                # Save to parquet file named by date
                filename = f"{date_key}.parquet"
                file_path = export_dir / filename

                if file_path.exists():
                    # Read existing data and append
                    existing_data = pq.read_table(file_path)
                    combined_data = pa.concat_tables([existing_data, daily_table])
                    # Sort by location_id and timestamp
                    combined_data = combined_data.sort_by(
                        [("location_id", "ascending"), ("timestamp", "ascending")]
                    )
                    key_columns = ["location_id", "timestamp", "network_name"]
                    # Remove duplicate rows
                    deduped_table = pa_ops.drop_duplicates(
                        combined_data, key_columns, keep="first"
                    )
                    pq.write_table(deduped_table, file_path, compression="BROTLI")
                    self.logger.info(
                        f"Added {daily_table.num_rows} availability records to existing file {filename}"
                    )
                else:
                    # Create new file
                    pq.write_table(daily_table, file_path, compression="BROTLI")
                    self.logger.info(
                        f"Created {filename} with {daily_table.num_rows} availability records"
                    )

            self.logger.info(
                f"Availability generation completed. Files saved to {export_dir}"
            )

        except Exception as e:
            self.logger.error(f"Error generating availability: {e}")

    def getProcessorClass(self) -> str:
        """Get the processor class used by the operator."""
        return self.processor_class

    def getProcessingSteps(self) -> list:
        """Get the processing steps configured for the operator."""
        return self.processing_steps