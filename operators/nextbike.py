import os
import glob
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import pyarrow_ops as pa_ops

from utils import DataPipelineLogger


class Nextbike:

    def __init__(
        self,
        input_data_dir_path,
        meta_data_dir_path,
        export_data_dir_path,
        logs_data_dir_path,
        processor_class,
        city_names=None,
        network_names=None,
        network_mappings=None,
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
            self.city_names = city_names or []
            self.network_names = network_names or []
            self.network_mappings = network_mappings or {}
            self.OSRM_ENABLED = osrm_enabled
            self.OSRM_ALTERNATIVE_PERCENTAGE = osrm_alternative_percentage
            self.db_connection = None
            self.processing_steps = processing_steps
            self.processor_class = processor_class

            # Setup logger
            self.logger = DataPipelineLogger.get_logger(
                name=self.__class__.__name__, log_file_path=self.log_file
            )

            self._initialize_database()

    def _create_network_mapping_table(self):
        """Create a DuckDB table for network mappings"""
        try:
            # Create a mapping table for network names
            mapping_data = {
                "network_name": list(self.network_mappings.keys()),
                "network_id": list(self.network_mappings.values()),
            }
            mapping_table = pa.table(mapping_data)

            # Register the table with DuckDB and create the table
            self.db_connection.register("temp_mapping_table", mapping_table)
            self.db_connection.execute(
                "CREATE OR REPLACE TABLE network_id_mappings AS SELECT * FROM temp_mapping_table"
            )
            self.db_connection.unregister("temp_mapping_table")

            self.logger.info("Network ID mapping table created successfully")
        except Exception as e:
            self.logger.error(f"Error creating network ID mapping table: {e}")

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

        # Collect paths for each data type
        bike_paths = []
        city_paths = []
        country_paths = []
        place_paths = []

        for date_dir in sorted(date_dirs):
            bikes_file = os.path.join(date_dir, "Bikes.parquet")
            cities_file = os.path.join(date_dir, "Cities.parquet")
            countries_file = os.path.join(date_dir, "Countries.parquet")
            places_file = os.path.join(date_dir, "Places.parquet")

            if os.path.exists(bikes_file):
                bike_paths.append(bikes_file)
            if os.path.exists(cities_file):
                city_paths.append(cities_file)
            if os.path.exists(countries_file):
                country_paths.append(countries_file)
            if os.path.exists(places_file):
                place_paths.append(places_file)

        # Connect to DuckDB
        con = duckdb.connect(database=f"./data/duckdb/nextbike.duckdb", read_only=False)
        # Store the connection for later use
        self.db_connection = con

        # Create tables/views
        self.logger.info("Creating filtered tables in DuckDB...")

        # Build filter conditions
        filter_conditions = self._build_filter_conditions(
            self.city_names, self.network_names
        )

        # Create filtered base tables that only load relevant data
        if filter_conditions:
            self.logger.info(
                f"Applying filters during data loading: {filter_conditions}"
            )

            # Create countries table first (needed for filtering other tables)
            if country_paths:
                countries_filter = ""
                if self.network_names:
                    network_list = ", ".join(
                        [f"'{name}'" for name in self.network_names]
                    )
                    countries_filter = f"WHERE name IN ({network_list})"

                con.execute(
                    f"""CREATE OR REPLACE TABLE countries AS 
                    SELECT DISTINCT ON (name, timestamp) * FROM read_parquet([{', '.join([f'\'{path}\'' for path in country_paths])}])
                    {countries_filter}"""
                )
                row_count = con.execute("SELECT COUNT(*) FROM countries").fetchone()[0]
                self.logger.info(f"countries table contains {row_count} rows")

            # Create cities table with filtering
            if city_paths:
                cities_filter_parts = []
                if self.city_names:
                    city_list = ", ".join([f"'{name}'" for name in self.city_names])
                    cities_filter_parts.append(f"c.name IN ({city_list})")
                if self.network_names:
                    network_list = ", ".join(
                        [f"'{name}'" for name in self.network_names]
                    )
                    cities_filter_parts.append(
                        f"c.parent_country_name IN ({network_list})"
                    )

                cities_filter = (
                    "WHERE " + " AND ".join(cities_filter_parts)
                    if cities_filter_parts
                    else ""
                )

                con.execute(
                    f"""CREATE OR REPLACE TABLE cities AS 
                    SELECT DISTINCT ON (c.uid, c.timestamp) c.* FROM read_parquet([{', '.join([f'\'{path}\'' for path in city_paths])}]) c
                    {cities_filter}"""
                )
                row_count = con.execute("SELECT COUNT(*) FROM cities").fetchone()[0]
                self.logger.info(f"cities table contains {row_count} rows")

            # Create places table with filtering based on cities
            if place_paths and city_paths:
                con.execute(
                    f"""CREATE OR REPLACE TABLE places AS 
                    SELECT DISTINCT ON (p.uid, p.timestamp) p.* FROM read_parquet([{', '.join([f'\'{path}\'' for path in place_paths])}]) p
                    WHERE EXISTS (
                        SELECT 1 FROM cities c 
                        WHERE p.city_uid = c.uid AND p.timestamp = c.timestamp
                    )"""
                )
                row_count = con.execute("SELECT COUNT(*) FROM places").fetchone()[0]
                self.logger.info(f"places table contains {row_count} rows")
            elif place_paths:
                con.execute(
                    f"CREATE OR REPLACE TABLE places AS SELECT DISTINCT ON (p.uid, p.timestamp) * FROM read_parquet([{', '.join([f'\'{path}\'' for path in place_paths])}])"
                )
                row_count = con.execute("SELECT COUNT(*) FROM places").fetchone()[0]
                self.logger.info(f"places table contains {row_count} rows")

            # Create bikes table with filtering based on places
            if bike_paths and place_paths:
                con.execute(
                    f"""CREATE OR REPLACE TABLE bikes AS 
                    SELECT DISTINCT ON (b.number, b.timestamp) b.* FROM read_parquet([{', '.join([f'\'{path}\'' for path in bike_paths])}]) b
                    WHERE EXISTS (
                        SELECT 1 FROM places p 
                        WHERE b.place_uid = p.uid AND b.timestamp = p.timestamp
                    )"""
                )
                row_count = con.execute("SELECT COUNT(*) FROM bikes").fetchone()[0]
                self.logger.info(f"bikes table contains {row_count} rows")
            elif bike_paths:
                con.execute(
                    f"CREATE OR REPLACE TABLE bikes AS SELECT DISTINCT ON (b.number, b.timestamp) * FROM read_parquet([{', '.join([f'\'{path}\'' for path in bike_paths])}])"
                )
                row_count = con.execute("SELECT COUNT(*) FROM bikes").fetchone()[0]
                self.logger.info(f"bikes table contains {row_count} rows")

        else:
            # No filters - load all data
            self.logger.info("No filters applied - loading all data")

            if country_paths:
                con.execute(
                    f"CREATE OR REPLACE TABLE countries AS SELECT DISTINCT * FROM read_parquet([{', '.join([f'\'{path}\'' for path in country_paths])}])"
                )
                row_count = con.execute("SELECT COUNT(*) FROM countries").fetchone()[0]
                self.logger.info(f"countries table contains {row_count} rows")

            if city_paths:
                con.execute(
                    f"CREATE OR REPLACE TABLE cities AS SELECT DISTINCT * FROM read_parquet([{', '.join([f'\'{path}\'' for path in city_paths])}])"
                )
                row_count = con.execute("SELECT COUNT(*) FROM cities").fetchone()[0]
                self.logger.info(f"cities table contains {row_count} rows")

            if place_paths:
                con.execute(
                    f"CREATE OR REPLACE TABLE places AS SELECT DISTINCT * FROM read_parquet([{', '.join([f'\'{path}\'' for path in place_paths])}])"
                )
                row_count = con.execute("SELECT COUNT(*) FROM places").fetchone()[0]
                self.logger.info(f"places table contains {row_count} rows")

            if bike_paths:
                con.execute(
                    f"CREATE OR REPLACE TABLE bikes AS SELECT DISTINCT * FROM read_parquet([{', '.join([f'\'{path}\'' for path in bike_paths])}])"
                )
                row_count = con.execute("SELECT COUNT(*) FROM bikes").fetchone()[0]
                self.logger.info(f"bikes table contains {row_count} rows")

        self._create_network_mapping_table()
        self.logger.info("DuckDB initialization completed successfully!")

        self._get_unique_location_coordinates()
        self.logger.info("Unique location coordinates extracted successfully")
        self._get_unique_station_names()
        self.logger.info("Unique station names extracted successfully")

    def _get_unique_location_coordinates(self):
        """Extract unique coordinates and append to location_coordinates.parquet"""
        try:
            # Define the meta data file path first
            meta_dir = Path(self.meta_data_dir_path)
            meta_dir.mkdir(parents=True, exist_ok=True)
            coordinates_file = meta_dir / "location_coordinates.parquet"

            # Get unique coordinates from vehicles table
            query = """
            SELECT DISTINCT ROUND(lat, 3) AS lat, ROUND(lng, 3) AS lng
            FROM places 
            WHERE lat IS NOT NULL AND lng IS NOT NULL 
            ORDER BY lat, lng
            """
            result = self.db_connection.execute(query).fetchall()

            if not result:
                self.logger.warning("No coordinates found in places table")
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
                    pq.write_table(
                        combined_data, coordinates_file, compression="BROTLI"
                    )
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

    def _get_unique_station_names(self):
        """Extract unique station names and append to station_names.parquet"""
        try:
            # Define the meta data file path first
            meta_dir = Path(self.meta_data_dir_path)
            meta_dir.mkdir(parents=True, exist_ok=True)
            station_names_file = meta_dir / "station_names.parquet"
            # Get unique station names from places table
            query = (
                "SELECT DISTINCT name FROM places WHERE name IS NOT NULL ORDER BY name"
            )
            result = self.db_connection.execute(query).fetchall()

            if not result:
                self.logger.warning("No station names found in places table")
                return

            # Extract stations from query result
            new_stations = [row[0] for row in result]

            # Handle existing file
            if station_names_file.exists():
                # Read existing data
                existing_data = pq.read_table(station_names_file)
                existing_stations = set(
                    existing_data.column("station_name").to_pylist()
                )
                max_station_name_id = max(
                    existing_data.column("station_name_id").to_pylist()
                )

                # Filter out station_names that already exist
                new_stations = [
                    name for name in new_stations if name not in existing_stations
                ]

                if not new_stations:
                    self.logger.info("No new station names to add")
                    self.db_connection.execute(
                        f"""
                        CREATE OR REPLACE TABLE station_names AS
                        SELECT * FROM parquet_scan('{station_names_file}')
                        """
                    )
                    return

                self.logger.info(
                    f"Adding {len(new_stations)} new station names to existing file"
                )
            else:
                max_station_name_id = 0
                self.logger.info(
                    f"Creating new station names file with {len(new_stations)} station names"
                )

            # Create data for new station names only
            if new_stations:
                new_station_name_ids = [
                    max_station_name_id + i + 1 for i in range(len(new_stations))
                ]

                new_data = pa.table(
                    {
                        "station_name_id": new_station_name_ids,
                        "station_name": new_stations,
                    }
                )

                # Append to existing file or create new one
                if station_names_file.exists():
                    existing_data = pq.read_table(station_names_file)
                    combined_data = pa.concat_tables([existing_data, new_data])
                    pq.write_table(
                        combined_data, station_names_file, compression="BROTLI"
                    )
                else:
                    pq.write_table(new_data, station_names_file, compression="BROTLI")

                self.logger.info(f"Saved station names to {station_names_file}")

            # Create DuckDB table from the parquet file
            self.db_connection.execute(
                f"""
                CREATE OR REPLACE TABLE station_names AS
                SELECT * FROM parquet_scan('{station_names_file}')
                """
            )

        except Exception as e:
            self.logger.error(f"Error extracting station names: {e}")

    def _build_filter_conditions(self, city_names=None, network_names=None):
        """Build SQL filter conditions for city and network names"""
        conditions = []

        if city_names and len(city_names) > 0:
            city_list = ", ".join([f"'{name}'" for name in city_names])
            conditions.append(f"c.name IN ({city_list})")

        if network_names and len(network_names) > 0:
            network_list = ", ".join([f"'{name}'" for name in network_names])
            conditions.append(f"co.name IN ({network_list})")

        return " AND " + " AND ".join(conditions) if conditions else ""

    def trip_generator(self):
        """Generate trips from the bikes table and save to daily parquet files"""
        ## die query generiert nur trips, wenn es zwischen zwei verschiedenen orten ist ODER, wenn sich der ort nicht geändert hat (jemand ist dumm im kreis gefahren, DABEI aber mindesten 5 minuten unterwegs war und innerhalb der 5 minuten eine zusätzliche messung war. dies entfernt fehler, wenn dateien fehlen, da sonst alles als ein trip zwischen den beiden messungen gewertet werden würde. [wenn man nur aufgrundlage der abwesenheit der bike id einen trip erstellt über eine gewisse zeit.])
        try:
            self.logger.info("Starting trip generation...")

            if self.city_names:
                self.logger.info(
                    f"Pre-filtered for cities: {', '.join(self.city_names)}"
                )
            if self.network_names:
                self.logger.info(
                    f"Pre-filtered for networks: {', '.join(self.network_names)}"
                )

            query = f"""
            WITH filtered_bikes AS (
                SELECT
                    b.number,
                    b.timestamp,
                    b.place_uid,
                    ROUND(p.lat, 3) AS lat,
                    ROUND(p.lng, 3) AS lng,
                    c.name AS city_name,
                    p.name AS place_name,
                    sn.station_name_id,
                    nm.network_id,
                    lc.location_id
                FROM bikes b
                JOIN places p ON b.place_uid = p.uid AND b.timestamp = p.timestamp
                JOIN cities c ON p.city_uid = c.uid AND b.timestamp = c.timestamp
                LEFT JOIN station_names sn ON p.name = sn.station_name
                LEFT JOIN network_id_mappings nm ON c.name = nm.network_name
                LEFT JOIN location_coordinates lc ON (ROUND(p.lat, 3) = lc.lat AND ROUND(p.lng, 3) = lc.lng)
            ),
            trips as (
                SELECT
                    *,
                    number AS vehicle_id,
                    CAST(to_timestamp(timestamp) AS TIMESTAMPTZ) AS timestamp_return,
                    place_uid AS station_id_return,
                    station_name_id AS station_name_id_return,
                    network_id AS network_name_return,
                    location_id AS location_id_return,
                    NULL::SMALLINT AS pedelec_battery_lend,
                    NULL::SMALLINT AS pedelec_battery_return,
                    NULL::INTEGER AS current_range_meters_lend,
                    NULL::INTEGER AS current_range_meters_return
                FROM (
                    SELECT
                        *,
                        CAST(to_timestamp(LAG(timestamp) OVER (PARTITION BY number ORDER BY timestamp)) AS TIMESTAMPTZ) AS timestamp_lend,
                        LAG(location_id) OVER (PARTITION BY number ORDER BY timestamp) AS prev_location_id,
                        LAG(timestamp) OVER (PARTITION BY number ORDER BY timestamp) AS prev_timestamp, 
                        LAG(place_uid) OVER (PARTITION BY number ORDER BY timestamp) AS station_id_lend,
                        LAG(station_name_id) OVER (PARTITION BY number ORDER BY timestamp) AS station_name_id_lend,
                        LAG(network_id) OVER (PARTITION BY number ORDER BY timestamp) AS network_name_lend,
                        LAG(location_id) OVER (PARTITION BY number ORDER BY timestamp) AS location_id_lend
                    FROM filtered_bikes
                )
                WHERE
                    prev_location_id IS NOT NULL
                    AND prev_location_id != location_id
                    AND station_id_lend != station_id_return
                    AND station_name_id_lend != station_name_id_return
                ORDER BY number, timestamp_lend
            )
            SELECT DISTINCT
                vehicle_id,
                timestamp_lend,
                timestamp_return,
                CAST(station_id_lend AS TEXT) AS station_id_lend,
                CAST(station_id_return AS TEXT) AS station_id_return,
                CAST(station_name_id_lend AS INTEGER) AS station_name_id_lend,
                CAST(station_name_id_return AS INTEGER) AS station_name_id_return,
                network_name_lend,
                network_name_return,
                CAST(location_id_lend AS INTEGER) AS location_id_lend,
                CAST(location_id_return AS INTEGER) AS location_id_return,
                pedelec_battery_lend,
                pedelec_battery_return,
                current_range_meters_lend,
                current_range_meters_return,
                FALSE AS predicted,
                DATE(timestamp_lend) AS trip_date
            FROM trips
            WHERE timestamp_lend IS NOT NULL
            ORDER BY vehicle_id, timestamp_lend;
            """

            result_query = self.db_connection.execute(query)
            result = result_query.fetch_arrow_table()

            if len(result) == 0:
                self.logger.warning("No trips generated")
                return

            self.logger.info(f"Generated {len(result)} trips")

            # Create export directory
            export_dir = Path(self.export_data_dir_path) / "trips"
            export_dir.mkdir(parents=True, exist_ok=True)

            # Group by trip_date and save to separate files
            trip_dates = result.column("trip_date").to_pylist()
            unique_dates = sorted(set(trip_dates))

            for date in unique_dates:
                # Filter rows for this date
                date_mask = pa.compute.equal(result.column("trip_date"), date)
                date_filtered = result.filter(date_mask)

                # Drop the trip_date column and ensure correct schema
                trips_table = date_filtered.select(
                    [
                        "vehicle_id",
                        "timestamp_lend",
                        "timestamp_return",
                        "station_id_lend",
                        "station_id_return",
                        "station_name_id_lend",
                        "station_name_id_return",
                        "network_name_lend",
                        "network_name_return",
                        "location_id_lend",
                        "location_id_return",
                        "pedelec_battery_lend",
                        "pedelec_battery_return",
                        "current_range_meters_lend",
                        "current_range_meters_return",
                        "predicted",
                    ]
                )

                # Save to parquet file named by date
                filename = f"{date}.parquet"
                file_path = export_dir / filename

                if file_path.exists():
                    # Read existing data and append
                    existing_data = pq.read_table(file_path)
                    combined_data = pa.concat_tables([existing_data, trips_table])
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
                        f"Added {len(trips_table)} trip records to existing file {filename}"
                    )
                else:
                    # Create new file
                    pq.write_table(trips_table, file_path, compression="BROTLI")
                    self.logger.info(
                        f"Created {filename} with {len(trips_table)} trip records"
                    )

            self.logger.info(f"Trip generation completed. Files saved to {export_dir}")

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
            if self.city_names:
                self.logger.info(
                    f"Using pre-filtered data for cities: {', '.join(self.city_names)}"
                )
            if self.network_names:
                self.logger.info(
                    f"Using pre-filtered data for networks: {', '.join(self.network_names)}"
                )

            # Create export directory for demand
            export_dir = Path(self.export_data_dir_path) / "demand"
            export_dir.mkdir(parents=True, exist_ok=True)

            # Load all trip files into a single table
            trip_file_paths = [str(f) for f in sorted(trip_files)]
            self.logger.info(
                f"Loading all {len(trip_file_paths)} trip files into single table..."
            )

            self.db_connection.execute("DROP TABLE IF EXISTS all_trips")
            paths_string = ", ".join([f"'{path}'" for path in trip_file_paths])
            self.db_connection.execute(
                f"CREATE TABLE all_trips AS SELECT * FROM read_parquet([{paths_string}])"
            )

            # Generate demand from all trips in one query
            query = """
            WITH rent_demand AS (
                SELECT 
                    location_id_lend AS location_id,
                    EXTRACT(EPOCH FROM timestamp_lend)::INTEGER AS timestamp,
                    network_name_lend AS network_name,
                    station_name_id_lend AS station_name_id,
                    station_id_lend AS station_id,
                    COUNT(*)::SMALLINT AS n_lends,
                    DATE(timestamp_lend) AS demand_date
                FROM all_trips
                WHERE location_id_lend IS NOT NULL AND timestamp_lend IS NOT NULL
                GROUP BY location_id_lend, EXTRACT(EPOCH FROM timestamp_lend)::INTEGER, network_name_lend, station_name_id_lend, station_id_lend, DATE(timestamp_lend)
            ),
            return_demand AS (
                SELECT 
                    location_id_return AS location_id,
                    EXTRACT(EPOCH FROM timestamp_return)::INTEGER AS timestamp,
                    network_name_return AS network_name,
                    station_name_id_return AS station_name_id,
                    station_id_return AS station_id,
                    COUNT(*)::SMALLINT AS n_returns,
                    DATE(timestamp_return) AS demand_date
                FROM all_trips
                WHERE location_id_return IS NOT NULL AND timestamp_return IS NOT NULL
                GROUP BY location_id_return, EXTRACT(EPOCH FROM timestamp_return)::INTEGER, network_name_return, station_name_id_return, station_id_return, DATE(timestamp_return)
            )
            SELECT DISTINCT
                COALESCE(r.location_id, re.location_id) AS location_id,
                CAST(to_timestamp(COALESCE(r.timestamp, re.timestamp)) AS TIMESTAMPTZ) AS timestamp,
                COALESCE(r.network_name, re.network_name) AS network_name,
                COALESCE(r.station_name_id, re.station_name_id) AS station_name_id,
                COALESCE(r.station_id, re.station_id) AS station_id,
                COALESCE(r.n_lends, 0::SMALLINT) AS n_lends,
                COALESCE(re.n_returns, 0::SMALLINT) AS n_returns,
                COALESCE(r.demand_date, re.demand_date) AS demand_date
            FROM rent_demand r
            FULL OUTER JOIN return_demand re
                ON r.location_id = re.location_id 
                AND r.timestamp = re.timestamp
                AND r.network_name = re.network_name
                AND r.station_name_id = re.station_name_id
                AND r.station_id = re.station_id
                AND r.demand_date = re.demand_date
            ORDER BY demand_date, location_id, timestamp;
            """

            self.logger.info("Generating demand from all trips...")
            result_query = self.db_connection.execute(query)
            result = result_query.fetch_arrow_table()

            # Clean up temporary table
            self.db_connection.execute("DROP TABLE all_trips")

            if len(result) == 0:
                self.logger.warning("No demand data generated")
                return

            self.logger.info(f"Generated {len(result)} total demand records")

            # Group by demand_date and save to separate files
            demand_dates = result.column("demand_date").to_pylist()
            unique_dates = sorted(set(demand_dates))

            for date in unique_dates:
                # Filter rows for this date
                date_mask = pa.compute.equal(result.column("demand_date"), date)
                date_filtered = result.filter(date_mask)

                # Drop the demand_date column
                demand_table = date_filtered.select(
                    [
                        "location_id",
                        "timestamp",
                        "network_name",
                        "station_name_id",
                        "station_id",
                        "n_lends",
                        "n_returns",
                    ]
                )

                # Save to parquet file named by date
                filename = f"{date}.parquet"
                file_path = export_dir / filename

                if file_path.exists():
                    # Read existing data and append
                    existing_data = pq.read_table(file_path)
                    combined_data = pa.concat_tables([existing_data, demand_table])
                    combined_data = combined_data.sort_by(
                        [("location_id", "ascending"), ("timestamp", "ascending")]
                    )
                    key_columns = ["location_id", "network_name", "timestamp"]
                    # make a distinction to avoid duplicate rows
                    deduped_table = pa_ops.drop_duplicates(
                        combined_data, key_columns, keep="first"
                    )
                    pq.write_table(deduped_table, file_path, compression="BROTLI")
                    self.logger.info(
                        f"Added {len(demand_table)} demand records to existing file {filename}"
                    )
                else:
                    # Create new file
                    pq.write_table(demand_table, file_path, compression="BROTLI")
                    self.logger.info(
                        f"Created {filename} with {len(demand_table)} demand records"
                    )

            self.logger.info(
                f"Demand generation completed. Files saved to {export_dir}"
            )

        except Exception as e:
            self.logger.error(f"Error generating demand: {e}")

    def availability_generator(self):
        """Generate bike availability data grouped by location and timestamp from raw data"""
        try:
            self.logger.info("Starting availability generation...")

            query = f"""
            WITH filtered_bikes AS (
                SELECT
                    b.number,
                    b.timestamp,
                    b.place_uid,
                    p.uid AS station_id,
                    p.lat,
                    p.lng,
                    p.name AS place_name,
                    c.name AS city_name
                FROM bikes b
                JOIN places p ON b.place_uid = p.uid AND b.timestamp = p.timestamp
                JOIN cities c ON p.city_uid = c.uid AND b.timestamp = c.timestamp
            ),
            availability_data AS (
                SELECT
                    CAST(lc.location_id AS INTEGER) AS location_id,
                    CAST(to_timestamp(fb.timestamp) AS TIMESTAMPTZ) AS timestamp,
                    nm.network_id AS network_name,
                    CAST(sn.station_name_id AS INTEGER) AS station_name_id,
                    CAST(fb.station_id AS TEXT) AS station_id,
                    COUNT(fb.number)::SMALLINT AS n_vehicles,
                    DATE(to_timestamp(fb.timestamp)) AS availability_date
                FROM filtered_bikes fb
                LEFT JOIN location_coordinates lc ON (ROUND(fb.lat, 3) = lc.lat AND ROUND(fb.lng, 3) = lc.lng)
                LEFT JOIN station_names sn ON fb.place_name = sn.station_name
                LEFT JOIN network_id_mappings nm ON fb.city_name = nm.network_name
                GROUP BY 
                    lc.location_id,
                    fb.timestamp,
                    fb.station_id,
                    sn.station_name_id,
                    nm.network_id
                ORDER BY availability_date, location_id, timestamp
            )
            SELECT DISTINCT
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
            if self.city_names:
                self.logger.info(
                    f"Pre-filtered for cities: {', '.join(self.city_names)}"
                )
            if self.network_names:
                self.logger.info(
                    f"Pre-filtered for networks: {', '.join(self.network_names)}"
                )

            result_query = self.db_connection.execute(query)
            result = result_query.fetch_arrow_table()

            if len(result) == 0:
                self.logger.warning("No availability data generated")
                return

            self.logger.info(f"Generated {len(result)} availability records")

            # Create export directory
            export_dir = Path(self.export_data_dir_path) / "availability"
            export_dir.mkdir(parents=True, exist_ok=True)

            # Group by availability_date and save to separate files
            availability_dates = result.column("availability_date").to_pylist()
            unique_dates = sorted(set(availability_dates))

            for date in unique_dates:
                # Filter rows for this date
                date_mask = pa.compute.equal(result.column("availability_date"), date)
                date_filtered = result.filter(date_mask)

                # Drop the availability_date column
                availability_table = date_filtered.select(
                    [
                        "location_id",
                        "timestamp",
                        "network_name",
                        "station_name_id",
                        "station_id",
                        "n_vehicles",
                    ]
                )

                # Save to parquet file named by date
                filename = f"{date}.parquet"
                file_path = export_dir / filename

                if file_path.exists():
                    # Read existing data and append
                    existing_data = pq.read_table(file_path)
                    combined_data = pa.concat_tables(
                        [existing_data, availability_table]
                    )
                    combined_data = combined_data.sort_by(
                        [("location_id", "ascending"), ("timestamp", "ascending")]
                    )
                    key_columns = ["location_id", "network_name", "timestamp"]
                    # make a distinction to avoid duplicate rows
                    deduped_table = pa_ops.drop_duplicates(
                        combined_data, key_columns, keep="first"
                    )

                    pq.write_table(deduped_table, file_path, compression="BROTLI")
                    self.logger.info(
                        f"Added {len(availability_table)} availability records to existing file {filename}"
                    )
                else:
                    # Create new file
                    pq.write_table(availability_table, file_path, compression="BROTLI")
                    self.logger.info(
                        f"Created {filename} with {len(availability_table)} availability records"
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
