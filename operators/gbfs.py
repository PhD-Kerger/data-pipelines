import os
import glob
import shutil
from random import randint

import requests
import duckdb
import tqdm
import pyarrow as pa
import numpy as np
import geopandas as gpd
import pyarrow.parquet as pq
import joblib
from geopy.distance import geodesic
from shapely.geometry import Point
from pathlib import Path
import tarfile

import pyarrow_ops as pa_ops

from utils import DataPipelineLogger


class GBFS:

    def __init__(
        self,
        input_data_dir_path,
        meta_data_dir_path,
        export_data_dir_path,
        logs_data_dir_path,
        rotating,
        processor_class,
        gbfs_version="3",
        network_names=None,
        network_mappings=None,
        operator=None,
        stations=False,
        osrm_enabled=False,
        osrm_alternative_percentage=0,
        processing_steps=["trips", "availability", "demand"],
    ):
        if not hasattr(self, "initialized"):
            self.initialized = True
            self.rotating = rotating
            self.gbfs_version = gbfs_version
            self.input_data_dir_path = input_data_dir_path
            self.meta_data_dir_path = meta_data_dir_path
            self.export_data_dir_path = export_data_dir_path
            self.log_file = Path(logs_data_dir_path) / "logs.log"
            self.network_names = network_names or []
            self.network_mappings = network_mappings or {}
            self.operator = operator
            self.stations = stations
            self.OSRM_ENABLED = osrm_enabled
            self.OSRM_ALTERNATIVE_PERCENTAGE = osrm_alternative_percentage
            self.db_connection = None
            self.processing_steps = processing_steps
            self.processor_class = processor_class

            if self.gbfs_version == "3":
                self.id_text = "vehicle_id"
            else:
                self.id_text = "bike_id"

            # Setup logger
            self.logger = DataPipelineLogger.get_logger(
                name=self.__class__.__name__, log_file_path=self.log_file
            )

            # Cache for geospatial data
            self._rheinland_pfalz_geometry = self._load_rheinland_pfalz_geometry()

            self._initialize_database()

    def _load_rheinland_pfalz_geometry(self):
        """Load the Rheinland-Pfalz shapefile for geospatial checks"""
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
            if self._rheinland_pfalz_geometry is None:
                return None

            point = Point(lng, lat)  # Note: shapely uses (lng, lat) order
            return self._rheinland_pfalz_geometry.contains(point)

        except Exception as e:
            self.logger.warning(
                f"Error checking point ({lat}, {lng}) in Rheinland-Pfalz: {e}"
            )
            return None

    def _adjust_network_name_based_on_geography(self, network_name, lat, lng):
        """Adjust network_name based on geographic location"""
        # Check if the point is in Rheinland-Pfalz
        in_rheinland_pfalz = self._is_point_in_rheinland_pfalz(lat, lng)

        if in_rheinland_pfalz is True:
            # Point is in Rheinland-Pfalz, so it's Ludwigshafen
            return "Ludwigshafen"
        elif in_rheinland_pfalz is False:
            # Point is not in Rheinland-Pfalz, so it's Mannheim (Baden-Württemberg)
            return "Mannheim"
        else:
            # Could not determine, keep original
            self.logger.warning(
                f"Could not determine location for point ({lat}, {lng}), keeping original network_name: {network_name}"
            )
            return network_name

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

    def _add_network_name(self):
        """Add city name to the vehicles table using SQL with Haversine distance calculation"""
        try:
            if self.operator == "voi":
                # Create a city reference table
                cities_sql = """
                CREATE OR REPLACE TABLE city_references AS
                SELECT * FROM VALUES
                    ('Karlsruhe', 48.9956, 8.4041),
                    ('Stuttgart', 48.7758, 9.1829),
                    ('Mannheim', 49.4875, 8.4660),
                    ('Pforzheim', 48.8958, 8.7039),
                    ('Reutlingen', 48.5211, 9.2036),
                    ('Tuebingen', 48.5204, 9.0582)
                AS cities(network_name, city_lat, city_lng)
                """
                self.db_connection.execute(cities_sql)

                # Add network_name column to vehicles table using SQL with Haversine distance
                update_sql = """
                ALTER TABLE vehicles_raw ADD COLUMN IF NOT EXISTS network_name VARCHAR;
                
                UPDATE vehicles_raw 
                SET network_name = (
                    SELECT network_name
                    FROM city_references
                    WHERE (
                        6371 * 2 * ASIN(SQRT(
                            POW(SIN(RADIANS(city_lat - vehicles_raw.lat) / 2), 2) +
                            COS(RADIANS(vehicles_raw.lat)) * COS(RADIANS(city_lat)) *
                            POW(SIN(RADIANS(city_lng - vehicles_raw.lon) / 2), 2)
                        ))
                    ) <= 20
                    ORDER BY (
                        6371 * 2 * ASIN(SQRT(
                            POW(SIN(RADIANS(city_lat - vehicles_raw.lat) / 2), 2) +
                            COS(RADIANS(vehicles_raw.lat)) * COS(RADIANS(city_lat)) *
                            POW(SIN(RADIANS(city_lng - vehicles_raw.lon) / 2), 2)
                        ))
                    ) ASC
                    LIMIT 1
                );
                
                -- Delete vehicles that don't have a network_name assigned
                DELETE FROM vehicles_raw WHERE network_name IS NULL;
                """
                self.db_connection.execute(update_sql)

                self.logger.info("City names added to vehicle data using SQL")
            elif (
                self.operator == "dott"
                and "Mannheim" in self.network_names
                and "Ludwigshafen" in self.network_names
            ):
                self.logger.info(
                    "Adjusting network names for Mannheim and Ludwigshafen based on geography"
                )
                # we need to determine based on geography if it is Mannheim or Ludwigshafen
                # First, add the network_name column if it doesn't exist
                update_sql = """
                ALTER TABLE vehicles_raw ADD COLUMN IF NOT EXISTS network_name VARCHAR;
                """
                self.db_connection.execute(update_sql)

                # Now, we need to fetch all rows and adjust network_name based on lat/lon
                select_sql = """
                SELECT rowid, lat, lon
                FROM vehicles_raw
                WHERE network_name IS NULL;
                """
                rows = self.db_connection.execute(select_sql).fetchall()

                if not rows:
                    self.logger.info("No rows to adjust - network_name already set")
                else:
                    self.logger.info(f"Adjusting {len(rows)} rows based on geography")

                    # Process in batches and use DuckDB's UPDATE directly
                    batch_size = 10000
                    for i in tqdm.tqdm(
                        range(0, len(rows), batch_size),
                        desc="Updating network names in batches",
                    ):
                        batch = rows[i : i + batch_size]

                        # Build WHEN conditions for batch
                        when_conditions = []
                        for rowid, lat, lon in batch:
                            if lat is not None and lon is not None:
                                adjusted_name = (
                                    self._adjust_network_name_based_on_geography(
                                        None, lat, lon
                                    )
                                )
                                when_conditions.append(
                                    f"WHEN rowid = {rowid} THEN '{adjusted_name}'"
                                )

                        if when_conditions:
                            case_statement = " ".join(when_conditions)

                            # Update the batch
                            update_query = f"""
                            UPDATE vehicles_raw
                            SET network_name = CASE
                                {case_statement}
                                ELSE network_name
                            END
                            WHERE rowid IN ({','.join([str(row[0]) for row in batch])});
                            """
                            self.db_connection.execute(update_query)

                self.logger.info("City names added to vehicle data based on geography")

            else:
                # City name can be used from first key of network_mappings
                network_name = list(self.network_mappings.keys())[0]
                update_sql = f"""
                ALTER TABLE vehicles_raw ADD COLUMN IF NOT EXISTS network_name VARCHAR;

                UPDATE vehicles_raw
                SET network_name = '{network_name}';
                """
                self.db_connection.execute(update_sql)

                self.logger.info(f"City name '{network_name}' added to vehicle data")

        except Exception as e:
            self.logger.error(f"Error adding city names: {e}")

    def _build_filter_conditions(self, network_names=None):
        """Build SQL filter conditions for network names"""
        conditions = []

        if network_names and len(network_names) > 0:
            network_list = ", ".join(
                [f"'{network_name}'" for network_name in network_names]
            )
            conditions.append(f"network_name IN ({network_list})")

        return " AND " + " AND ".join(conditions) if conditions else ""

    def _add_station_name(self):
        """Add station names to the vehicles_raw table by joining on the station_id field"""
        try:
            # Use the existing station_information table that was created
            # Add station_name column to vehicles table using SQL
            update_sql = """
            ALTER TABLE vehicles_raw ADD COLUMN IF NOT EXISTS station_name VARCHAR;

            UPDATE vehicles_raw
            SET station_name = (
                SELECT station_name
                FROM station_information
                WHERE station_information.station_id = vehicles_raw.station_id
                LIMIT 1
            )
            WHERE vehicles_raw.station_id IS NOT NULL;
            """
            self.db_connection.execute(update_sql)

            self.logger.info("Station names added to vehicle data using SQL")
        except Exception as e:
            self.logger.error(f"Error adding station names: {e}")

    def _overwrite_vehicle_id(self):
        """Overwrites the vehicle id (if possible) with a computed value from another field to allow later trip generation."""
        if self.operator == "voi":
            # The rental_uri_ios looks like this: https://lqfa.adj.st/closest_vehicle?adj_t=b2hnabv&adj_deep_link=voiapp%3A%2F%2Fscooter%2Fnml8&adj_campaign=nvbw.de.
            # We can extract everyhing between "scooter%2" and "&adj_campaign" and set this as the new vehicle_id
            self.db_connection.execute(
                f"""
                UPDATE vehicles_raw
                SET {self.id_text} = regexp_extract(rental_uri_ios, 'scooter%2F([^&]+)', 1)
                WHERE rental_uri_ios LIKE '%scooter%2F%'
                """
            )
        elif self.operator == "dott":
            # remove DOC:Vehicle: prefix from vehicle_id if present. MobidataBW feed has this prefix, official Dott GBFS does not have it.
            # Results in different vehicle ids for same vehicle which break distinct statements.
            self.db_connection.execute(
                f"""
                UPDATE vehicles_raw
                SET {self.id_text} = REPLACE({self.id_text}, 'DOC:Vehicle:', '')
                WHERE {self.id_text} LIKE 'DOC:Vehicle:%'
                """
            )
            # remove DOD:Vehicle: prefix from vehicle_id if present. MobidataBW feed has this prefix, official Dott GBFS does not have it.
            # Results in different vehicle ids for same vehicle which break distinct statements.
            self.db_connection.execute(
                f"""
                UPDATE vehicles_raw
                SET {self.id_text} = REPLACE({self.id_text}, 'DOD:Vehicle:', '')
                WHERE {self.id_text} LIKE 'DOD:Vehicle:%'
                """
            )
            # remove DOG:Vehicle: prefix from vehicle_id if present. MobidataBW feed has this prefix, official Dott GBFS does not have it.
            self.db_connection.execute(
                f"""
                UPDATE vehicles_raw
                SET {self.id_text} = REPLACE({self.id_text}, 'DOG:Vehicle:', '')
                WHERE {self.id_text} LIKE 'DOG:Vehicle:%'
                """
            )
            # remove DOJ:Vehicle: prefix from vehicle_id if present. MobidataBW feed has this prefix, official Dott GBFS does not have it.
            self.db_connection.execute(
                f"""
                UPDATE vehicles_raw
                SET {self.id_text} = REPLACE({self.id_text}, 'DOJ:Vehicle:', '')
                WHERE {self.id_text} LIKE 'DOJ:Vehicle:%'
                """
            )
        elif self.operator == "bolt":
            # Bolt does not have anything that we can use
            pass
        elif self.operator == "lime":
            # "vehicle_id": "LMG:Vehicle:V4DBK7TZMQS2M": we remove the suffix LMG:Vehicle:
            self.db_connection.execute(
                f"""
                UPDATE vehicles_raw
                SET {self.id_text} = REPLACE({self.id_text}, 'LMG:Vehicle:', '')
                WHERE {self.id_text} LIKE 'LMG:Vehicle:%'
                """
            )
        elif self.operator == "regioradstuttgart":
            self.logger.info(
                "Overwriting vehicle IDs for RegioRadStuttgart based on rental_uri_ios"
            )
            # The rental_uri_ios looks like this: https://www.callabike.de/bike?number=12583
            # or https://www.regioradstuttgart.de/bike?number=13664
            # We can extract the number parameter in the end and set this as the new vehicle_id
            self.db_connection.execute(
                f"""
                UPDATE vehicles_raw
                SET {self.id_text} = regexp_extract(rental_uri_ios, 'number=([0-9]+)', 1)
                WHERE rental_uri_ios LIKE 'https://www.callabike.de/bike?number=%'
                   OR rental_uri_ios LIKE 'https://www.regioradstuttgart.de/bike?number=%'
                """
            )
        elif self.operator == "callabike":
            # The rental_uri_ios looks like this: https://www.callabike.de/bike?number=12583
            # We can extract the number parameter in the end and set this as the new vehicle_id
            self.db_connection.execute(
                f"""
                UPDATE vehicles_raw
                SET {self.id_text} = regexp_extract(rental_uri_ios, 'number=([0-9]+)', 1)
                WHERE rental_uri_ios LIKE 'https://www.callabike.de/bike?number=%'
                """
            )
        elif self.operator == "zeus":
            # The rental uri web looks like this: "https://zeus.city/api/v1/applinks?vehicle=5867"
            # We can extract the vehicle parameter in the end and set this as the new vehicle_id
            self.db_connection.execute(
                f"""
                UPDATE vehicles_raw
                SET {self.id_text} = regexp_extract(rental_uri_web, 'vehicle=([0-9]+)', 1)
                WHERE rental_uri_web LIKE 'https://zeus.city/api/v1/applinks?vehicle=%'
                """
            )

    def _overwrite_station_id(self):
        """Overwrites the station id with a shorter integer id in the vehicles raw table with station_id_mappings table"""
        self.db_connection.execute(
            f"""
            UPDATE vehicles_raw
            SET station_id = (
                SELECT new_station_id
                FROM station_id_mappings
                WHERE station_id_mappings.station_id = vehicles_raw.station_id
            )
            WHERE vehicles_raw.station_id IS NOT NULL;
            """
        )

    def _initialize_database(self):
        """Initialize DuckDB database with views pointing to Parquet files"""
        # Set the data directory path
        data_dir = Path(__file__).parent.parent / self.input_data_dir_path
        if not data_dir.exists():
            self.logger.error(f"Data directory {data_dir} does not exist.")
            return

        if self.gbfs_version == "3":
            gbfs_file_name = "vehicle_status"
        else:
            gbfs_file_name = "free_bike_status"

            # Find all date directories in yyyy-mm-dd format
        date_pattern = os.path.join(data_dir, gbfs_file_name, "????-??-??")
        date_dirs = glob.glob(date_pattern)

        if not date_dirs:
            self.logger.warning(f"No date directories found in {data_dir}")
            return

        # Collect paths for free bike status files
        vehicle_paths = []

        for date_dir in sorted(date_dirs):
            vehicles_file = os.path.join(
                gbfs_file_name, date_dir, f"{gbfs_file_name}.parquet"
            )

            if os.path.exists(vehicles_file):
                vehicle_paths.append(vehicles_file)

        if not vehicle_paths:
            self.logger.warning(f"No vehicle status files found in {date_dir}")
            return

        # Remove duckdb directory if it exists
        if os.path.exists("./data/duckdb"):
            shutil.rmtree("./data/duckdb", ignore_errors=True)

        # Connect to DuckDB
        os.makedirs(f"./data/duckdb", exist_ok=True)

        con = duckdb.connect(
            database=f"./data/duckdb/{self.operator}.duckdb", read_only=False
        )
        con.execute("SET memory_limit='100GB'")

        # Store the connection for later use
        self.db_connection = con

        # Create tables/views
        self.logger.info("Creating tables in DuckDB...")
        # Create base view that points to the Parquet files
        if vehicle_paths:
            earliest_date = min(date_dirs).split("/")[-1]
            self.logger.info(f"Earliest date found: {earliest_date}")
            # timestamp,last_updated,vehicle_id,lat,lon,rental_uri_android,last_reported,current_range_meters,current_fuel_percent,station_id
            limit_clause = f" LIMIT 100000000"
            con.execute(
                f"""CREATE OR REPLACE TABLE vehicles_raw AS SELECT 
                    timestamp,
                    last_updated,
                    {self.id_text} AS vehicle_id,
                    ROUND(lat, 3) AS lat,
                    ROUND(lon, 3) AS lon,
                    rental_uri_ios,
                    last_reported,
                    current_range_meters,
                    current_fuel_percent,
                    station_id
                FROM read_parquet([{', '.join([f'\'{path}\'' for path in vehicle_paths])}]){limit_clause}"""
            )

            # check if last_reported column has NaT value. If you find one, log a warning and skip filtering.
            result = con.execute(
                "SELECT COUNT(*) FROM vehicles_raw WHERE last_reported IS NULL"
            ).fetchone()
            if result[0] > 0:
                self.logger.warning(
                    f"Found {result[0]} rows with NaT in last_reported column.. Skipping filtering by last_reported."
                )
            else:
                # filter all out that are before earliest date
                con.execute(
                    f"CREATE OR REPLACE TABLE vehicles_raw AS SELECT * FROM vehicles_raw WHERE last_reported >= '{earliest_date}'"
                )
                self.logger.info(
                    f"Filtered vehicles_raw to only include data from {earliest_date} onwards based on last_reported column."
                )

            self.logger.info(
                f"Created vehicles_raw table with {len(vehicle_paths)} files"
            )
            # print number of rows in vehicles_raw
            row_count = con.execute("SELECT COUNT(*) FROM vehicles_raw").fetchone()[0]
            self.logger.info(f"vehicles_raw contains {row_count} rows")

        # Add the city name as a new column to the vehicles_raw. Can then be used to filter data.
        # For instance, Voi has multiple cities in one feed, there this is needed.
        self._add_network_name()
        station_paths = None
        if self.stations:
            station_paths = []
            # Station names are in a separate JSON from the station_information GBFS feed.
            # Collect paths for station information files
            for date_dir in sorted(date_dirs):
                tar_file_path = os.path.join(
                    "station_information", f"{date_dir}.tar.gz"
                ).replace(gbfs_file_name, "station_information")

                if os.path.exists(tar_file_path):
                    try:
                        # Extract the tar.gz file
                        with tarfile.open(tar_file_path, "r:gz") as tar:
                            os.makedirs(
                                tar_file_path.replace(".tar.gz", "").replace(
                                    "station_information", "tmp"
                                ),
                                exist_ok=True,
                            )
                            tar.extractall(
                                path=tar_file_path.replace(".tar.gz", "").replace(
                                    "station_information", "tmp"
                                )
                            )

                        extracted_dir = tar_file_path.replace(".tar.gz", "").replace(
                            "station_information", "tmp"
                        )
                        json_files = [
                            os.path.join(extracted_dir, f)
                            for f in os.listdir(extracted_dir)
                            if f.endswith(".json")
                        ]

                        if json_files:
                            station_paths.extend(json_files)
                        else:
                            self.logger.warning(
                                f"No JSON files found in {tar_file_path}"
                            )

                    except Exception as e:
                        self.logger.error(f"Error extracting {tar_file_path}: {e}")
                        tmp_dir = tar_file_path.replace(".tar.gz", "").replace(
                            "station_information", "tmp"
                        )
                        # if os.path.exists(tmp_dir):
                        #     os.rmdir(tmp_dir)
                else:
                    self.logger.warning(f"Tar file {tar_file_path} does not exist")

        if station_paths:
            # We need a table that has a mapping between station_id and name for join
            self._create_station_information_table(station_paths)
            # We need a table that only has the station names with an id (for later replacement)
            self._create_station_names_table()
            # We overwrite the old station name with the new one
            self._add_station_name()
            # We get all available station ids and add a mapping to shorter ones
            self._get_unique_station_ids()
            # We overwrite the old station ids with the new ones
            self._overwrite_station_id()
            self.logger.info("Station IDs overwritten successfully")

        # Vehicle IDs normally rotate in GBFS. But there are sometimes fields that allow to identify
        # ad vehicle nevertheless.
        self._overwrite_vehicle_id()

        # Build filter conditions
        filter_conditions = self._build_filter_conditions(self.network_names)

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

        # Create network mapping table
        self._create_network_mapping_table()

        self._get_unique_location_coordinates()
        self.logger.info("Unique location coordinates extracted and saved")

        self.logger.info("DuckDB initialization completed successfully!")

    def _create_station_information_table(self, station_paths):
        """Extract station information from JSON files and create station_information table"""
        try:
            if self.gbfs_version == "3":
                station_select = "unnest.name[1].text::TEXT AS station_name"
            else:
                station_select = "unnest.name::TEXT AS station_name"

            if self.operator == "regioradstuttgart" or self.operator == "callabike":
                # RegioRadStuttgart and CallABike have station ids with prefix CAB:Station: that need to be removed,
                # else we will have duplicates later on.
                self.db_connection.execute(
                    f"""
                    CREATE OR REPLACE TABLE station_information AS
                        WITH extracted_stations AS (
                            SELECT DISTINCT
                                REPLACE(unnest.station_id::TEXT, 'CAB:Station:', '') AS station_id,
                                {station_select}
                            FROM read_json([{', '.join([f"'{path}'" for path in station_paths])}]),
                            UNNEST(data.stations) AS unnest
                        )
                        SELECT DISTINCT station_id, station_name
                        FROM extracted_stations
                        WHERE station_id IS NOT NULL AND station_name IS NOT NULL
                    """
                )
                self.logger.info(
                    f"Station information table created successfully for {self.operator}"
                )
            else:
                # Default case for all other operators
                self.db_connection.execute(
                    f"""
                    CREATE OR REPLACE TABLE station_information AS
                        WITH extracted_stations AS (
                            SELECT DISTINCT
                                unnest.station_id::TEXT AS station_id,
                                {station_select}
                            FROM read_json([{', '.join([f"'{path}'" for path in station_paths])}]),
                            UNNEST(data.stations) AS unnest
                        )
                        SELECT DISTINCT station_id, station_name
                        FROM extracted_stations
                        WHERE station_id IS NOT NULL AND station_name IS NOT NULL
                    """
                )
                self.logger.info(
                    "Default station information table created successfully"
                )

        except Exception as e:
            self.logger.error(f"Error extracting station information: {e}")

    def _create_station_names_table(self):
        """Create station_names table with incremental IDs from station_information"""
        try:
            # Define the meta data file path
            meta_dir = Path(self.meta_data_dir_path)
            meta_dir.mkdir(parents=True, exist_ok=True)
            station_names_file = meta_dir / "station_names.parquet"

            # Get unique station names from station_information table
            query = """
            SELECT DISTINCT station_name
            FROM station_information
            WHERE station_name IS NOT NULL
            ORDER BY station_name
            """
            result = self.db_connection.execute(query).fetchall()

            if not result:
                self.logger.warning(
                    "No station names found in station_information table"
                )
                return

            # Extract station names from query result
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
            self.logger.error(f"Error creating station names table: {e}")

    def _get_unique_location_coordinates(self):
        """Extract unique coordinates and append to location_coordinates.parquet"""
        try:
            # Define the meta data file path first
            meta_dir = Path(self.meta_data_dir_path)
            meta_dir.mkdir(parents=True, exist_ok=True)
            coordinates_file = meta_dir / "location_coordinates.parquet"

            # Get unique coordinates from vehicles table
            query = """
            SELECT DISTINCT lat, lon AS lng
            FROM vehicles 
            WHERE lat IS NOT NULL AND lon IS NOT NULL 
            ORDER BY lat, lng
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

    def _get_unique_station_ids(self):
        """Get unique station IDs from the vehicles table"""
        try:
            # Define the meta data file path first
            meta_dir = Path(self.meta_data_dir_path)
            meta_dir.mkdir(parents=True, exist_ok=True)
            station_ids_file = meta_dir / "station_ids.parquet"

            # Get unique station ids from vehicles table
            query = """
            SELECT DISTINCT station_id
            FROM vehicles_raw
            WHERE station_id IS NOT NULL
            ORDER BY station_id
            """
            result = self.db_connection.execute(query).fetchall()

            if not result:
                self.logger.warning("No station IDs found in vehicles table")
                return

            # Extract station ids from query result
            new_station_ids = [row[0] for row in result]

            # Handle existing file
            if station_ids_file.exists():
                # Read existing data
                existing_data = pq.read_table(station_ids_file)
                existing_station_ids = set(
                    existing_data.column("station_id").to_pylist()
                )

                # Filter out station ids that already exist
                new_station_ids = [
                    station_id
                    for station_id in new_station_ids
                    if station_id not in existing_station_ids
                ]
                max_new_station_id = max(
                    existing_data.column("new_station_id").to_pylist()
                )

                if not new_station_ids:
                    self.logger.info("No new station IDs to add")
                    # Still create the DuckDB table from existing file
                    self.db_connection.execute(
                        f"""
                        CREATE OR REPLACE TABLE station_id_mappings AS
                        SELECT * FROM parquet_scan('{station_ids_file}')
                        """
                    )
                    return

                self.logger.info(
                    f"Adding {len(new_station_ids)} new station IDs to existing file"
                )
            else:
                max_new_station_id = 0
                self.logger.info(
                    f"Creating new station IDs file with {len(new_station_ids)} station IDs"
                )

            # Create data for new station IDs only
            if new_station_ids:
                new_station_ids_new = [
                    max_new_station_id + i + 1 for i in range(len(new_station_ids))
                ]
                station_ids = new_station_ids

                new_data = pa.table(
                    {"new_station_id": new_station_ids_new, "station_id": station_ids}
                )

                # Append to existing file or create new one
                if station_ids_file.exists():
                    existing_data = pq.read_table(station_ids_file)
                    combined_data = pa.concat_tables([existing_data, new_data])
                    pq.write_table(
                        combined_data, station_ids_file, compression="BROTLI"
                    )
                else:
                    pq.write_table(new_data, station_ids_file, compression="BROTLI")

                self.logger.info(f"Saved station IDs to {station_ids_file}")

            # Create DuckDB table from the parquet file
            self.db_connection.execute(
                f"""
                CREATE OR REPLACE TABLE station_id_mappings AS
                SELECT * FROM parquet_scan('{station_ids_file}')
                """
            )

        except Exception as e:
            self.logger.error(f"Error extracting station IDs: {e}")

    def _build_filter_conditions(self, network_names=None):
        """Build SQL filter conditions for network names"""
        conditions = []

        if network_names and len(network_names) > 0:
            network_list = ", ".join(
                [f"'{network_name}'" for network_name in network_names]
            )
            conditions.append(f"network_name IN ({network_list})")

        return " AND " + " AND ".join(conditions) if conditions else ""

    def _calculate_distance(self, lon_lend, lat_lend, lon_return, lat_return):
        # Calculate geospatial and range-based features
        if self.OSRM_ENABLED:
            response = requests.get(
                f"http://localhost:5000/route/v1/bike/{lon_lend},{lat_lend};{lon_return},{lat_return}?overview=false&alternatives=3"
            )
            data = response.json()
            # check if to use alternative with a certain probability
            if "routes" in data and len(data["routes"]) > 0:
                rand = randint(1, 100)
                if rand <= self.OSRM_ALTERNATIVE_PERCENTAGE:
                    distance = int(data["routes"][-1]["distance"])
                else:
                    distance = int(data["routes"][0]["distance"])
            else:
                # Use geopy for geodesic distance as fallback
                distance = int(
                    geodesic((lat_lend, lon_lend), (lat_return, lon_return)).meters
                )
        else:
            # Use geopy for geodesic distance
            distance = int(
                geodesic((lat_lend, lon_lend), (lat_return, lon_return)).meters
            )

        return distance

    def trip_generator(self):
        """Generate trips from the vehicles table and save to daily parquet files"""
        try:
            result = []
            self.logger.info("Starting trip generation...")
            if self.rotating:
                # Generate trips with Machine Learning for rotating vehicle ids.
                # Results for that are available in the Trip Destination Prediction Paper.
                # For that, we need to iterate over all cities and make it city by city.
                # If you have further cities, add them as needed.
                for network_name in self.network_names:
                    self.logger.info(
                        f"Generating trips for city: {network_name} with ML model"
                    )
                    if "Mannheim" == network_name:
                        model = joblib.load(
                            "./data/internal/machine-learning/xgboost-models/mannheim.pkl"
                        )
                        scaler = joblib.load(
                            "./data/internal/machine-learning/scaler/mannheim.pkl"
                        )
                    elif "Stuttgart" == network_name:
                        model = joblib.load(
                            "./data/internal/machine-learning/xgboost-models/stuttgart.pkl"
                        )
                        scaler = joblib.load(
                            "./data/internal/machine-learning/scaler/stuttgart.pkl"
                        )
                    elif "Ludwigshafen" == network_name:
                        model = joblib.load(
                            "./data/internal/machine-learning/xgboost-models/ludwigshafen.pkl"
                        )
                        scaler = joblib.load(
                            "./data/internal/machine-learning/scaler/ludwigshafen.pkl"
                        )
                    elif "Heidelberg" == network_name:
                        model = joblib.load(
                            "./data/internal/machine-learning/xgboost-models/heidelberg.pkl"
                        )
                        scaler = joblib.load(
                            "./data/internal/machine-learning/scaler/heidelberg.pkl"
                        )
                    elif "Karlsruhe" == network_name:
                        model = joblib.load(
                            "./data/internal/machine-learning/xgboost-models/karlsruhe.pkl"
                        )
                        scaler = joblib.load(
                            "./data/internal/machine-learning/scaler/karlsruhe.pkl"
                        )
                    else:
                        self.logger.warning(
                            f"No model available for {network_name}. Skipped."
                        )
                        continue

                    trip_matching = f"""WITH first_use AS (
                        SELECT DISTINCT ON (v.{self.id_text})
                            v.{self.id_text} as vehicle_id,
                            (v.current_fuel_percent * 100)::SMALLINT AS pedelec_battery,
                            v.current_range_meters::INTEGER AS current_range_meters,
                            lc.location_id, 
                            lc.lng,
                            lc.lat,
                            epoch(v.last_updated) AS timestamp,
                            v.network_name AS network_name
                        FROM vehicles v
                        LEFT JOIN location_coordinates lc 
                            ON (v.lon = lc.lng 
                            AND v.lat = lc.lat)
                        WHERE v.network_name = '{network_name}'
                        ORDER BY v.{self.id_text}, v.timestamp ASC
                        ),
                        last_use AS (
                        SELECT DISTINCT ON (v.{self.id_text})
                            v.{self.id_text} as vehicle_id,
                            (v.current_fuel_percent * 100)::SMALLINT AS pedelec_battery,
                            v.current_range_meters::INTEGER AS current_range_meters,
                            lc.location_id,
                            lc.lng,
                            lc.lat,
                            epoch(v.last_updated) AS timestamp,
                            v.network_name AS network_name
                        FROM vehicles v
                        LEFT JOIN location_coordinates lc 
                            ON (v.lon = lc.lng 
                            AND v.lat = lc.lat)
                        WHERE v.network_name = '{network_name}'
                        ORDER BY v.{self.id_text}, v.timestamp DESC
                        )
                        SELECT
                            fu.vehicle_id AS vehicle_id_lend,
                            lu.vehicle_id AS vehicle_id_return,
                            fu.pedelec_battery AS pedelec_battery_lend,
                            lu.pedelec_battery AS pedelec_battery_return,
                            fu.current_range_meters AS current_range_meters_lend,
                            lu.current_range_meters AS current_range_meters_return,
                            fu.location_id AS location_id_lend,
                            lu.location_id AS location_id_return,
                            fu.lng AS lng_lend,
                            fu.lat AS lat_lend,
                            lu.lng AS lng_return,
                            lu.lat AS lat_return,
                            fu.timestamp AS timestamp_lend,
                            lu.timestamp AS timestamp_return,
                            DATE(timezone('UTC', to_timestamp(fu.timestamp))) AS trip_date,
                            fu.network_name,
                            (lu.timestamp - fu.timestamp) AS time_diff,
                            (fu.pedelec_battery - lu.pedelec_battery) AS battery_diff,
                            (fu.current_range_meters - lu.current_range_meters) AS range_diff
                        FROM first_use fu
                        CROSS JOIN last_use lu
                        WHERE
                            (lu.timestamp - fu.timestamp) <= 1800
                            AND (lu.timestamp - fu.timestamp) > 0 
                            AND (fu.pedelec_battery - lu.pedelec_battery) BETWEEN 0 AND 10
                        ORDER BY fu.vehicle_id, fu.timestamp ASC;
                    """

                    trip_matches_query = self.db_connection.execute(trip_matching)
                    trip_matches = trip_matches_query.fetchall()
                    if not trip_matches:
                        self.logger.warning(f"No trip matches found for {network_name}")
                        continue
                    self.logger.info(
                        f"Found {len(trip_matches)} trip matches for {network_name}"
                    )

                    blacklist = []
                    destination_candidate_pairings = {}

                    # calculate distance for each trip_match. remove trip match, if exceeds 3000m
                    for match in trip_matches:
                        match = {
                            "vehicle_id_lend": match[0],
                            "vehicle_id_return": match[1],
                            "pedelec_battery_lend": match[2],
                            "pedelec_battery_return": match[3],
                            "current_range_meters_lend": match[4],
                            "current_range_meters_return": match[5],
                            "location_id_lend": match[6],
                            "location_id_return": match[7],
                            "lng_lend": match[8],
                            "lat_lend": match[9],
                            "lng_return": match[10],
                            "lat_return": match[11],
                            "timestamp_lend": int(match[12]),
                            "timestamp_return": int(match[13]),
                            "trip_date": match[14],
                            "network_name": match[15],
                            "time_diff": int(match[16]),
                            "battery_diff": match[17],
                            "range_diff": match[18],
                        }
                        if match["vehicle_id_lend"] in blacklist:
                            continue

                        # R6:
                        distance = self._calculate_distance(
                            match["lng_lend"],
                            match["lat_lend"],
                            match["lng_return"],
                            match["lat_return"],
                        )
                        if distance > 3000:
                            continue

                        mean_speed = (
                            round((distance / 1000) / (match["time_diff"] / 3600), 2)
                            if match["time_diff"] > 0
                            else 0
                        )
                        mean_speed_range_based = (
                            round(
                                (match["range_diff"] / 1000)
                                / (match["time_diff"] / 3600),
                                2,
                            )
                            if match["time_diff"] > 0
                            else 0
                        )

                        # R5:
                        if mean_speed > 20:
                            continue

                        # R7:
                        if (
                            match["time_diff"] <= 6
                            and match["range_diff"] == 0
                            and match["battery_diff"] <= 1
                            and distance < 1000
                        ):
                            destination_candidate_pairings.setdefault(
                                match["trip_date"], {}
                            ).setdefault(match["vehicle_id_lend"], [])
                            destination_candidate_pairings[match["trip_date"]][
                                match["vehicle_id_lend"]
                            ] = [
                                {
                                    "vehicle_id_rental": match["vehicle_id_lend"],
                                    "vehicle_id_return": match["vehicle_id_return"],
                                    "lat_rental": match["lat_lend"],
                                    "lng_rental": match["lng_lend"],
                                    "lat_return": match["lat_return"],
                                    "lng_return": match["lng_return"],
                                    "location_id_rental": match["location_id_lend"],
                                    "location_id_return": match["location_id_return"],
                                    "timestamp_rental": match["timestamp_lend"],
                                    "timestamp_return": match["timestamp_return"],
                                    "battery_rental": match["pedelec_battery_lend"],
                                    "battery_return": match["pedelec_battery_return"],
                                    "range_rental": match["current_range_meters_lend"],
                                    "range_return": match[
                                        "current_range_meters_return"
                                    ],
                                    "distance": distance,
                                    "time_diff": match["time_diff"],
                                    "battery_diff": match["battery_diff"],
                                    "range_diff": match["range_diff"],
                                    "speed": mean_speed,
                                    "speed_range_based": mean_speed_range_based,
                                    "date": match["trip_date"],
                                }
                            ]
                            blacklist.append(match["vehicle_id_lend"])
                        else:
                            destination_candidate_pairings.setdefault(
                                match["trip_date"], {}
                            ).setdefault(match["vehicle_id_lend"], []).append(
                                {
                                    "vehicle_id_rental": match["vehicle_id_lend"],
                                    "vehicle_id_return": match["vehicle_id_return"],
                                    "lat_rental": match["lat_lend"],
                                    "lng_rental": match["lng_lend"],
                                    "lat_return": match["lat_return"],
                                    "lng_return": match["lng_return"],
                                    "location_id_rental": match["location_id_lend"],
                                    "location_id_return": match["location_id_return"],
                                    "timestamp_rental": match["timestamp_lend"],
                                    "timestamp_return": match["timestamp_return"],
                                    "battery_rental": match["pedelec_battery_lend"],
                                    "battery_return": match["pedelec_battery_return"],
                                    "range_rental": match["current_range_meters_lend"],
                                    "range_return": match[
                                        "current_range_meters_return"
                                    ],
                                    "distance": distance,
                                    "time_diff": match["time_diff"],
                                    "battery_diff": match["battery_diff"],
                                    "range_diff": match["range_diff"],
                                    "speed": mean_speed,
                                    "speed_range_based": mean_speed_range_based,
                                    "date": match["trip_date"],
                                }
                            )

                    # Predict with ML model the most likely return for each rental
                    self.logger.info(
                        f"Predicting most likely return for each rental in {network_name}"
                    )

                    for _, rentals in tqdm.tqdm(
                        destination_candidate_pairings.items(),
                        desc=f"Processing dates for {network_name}",
                        unit="date",
                        total=len(destination_candidate_pairings),
                    ):
                        for rental in rentals:
                            distances = []
                            for ret in rentals[rental]:
                                ret_features_array = np.array(
                                    [
                                        [
                                            ret["time_diff"],
                                            ret["battery_diff"],
                                            ret["range_diff"],
                                            ret["distance"],
                                            ret["speed"],
                                            ret["speed_range_based"],
                                        ]
                                    ]
                                )

                                # Scale the features and predict
                                X = scaler.transform(ret_features_array)
                                feature_vector = np.array(
                                    [
                                        [ret["lat_rental"], ret["lng_rental"]]
                                        + list(X[0])
                                    ]
                                )
                                y_pred = model.predict(feature_vector)

                                # Calculate geodesic distances for evaluation
                                distance = geodesic(
                                    (y_pred[0][0], y_pred[0][1]),
                                    (ret["lat_return"], ret["lng_return"]),
                                    ellipsoid="WGS-84",
                                ).m
                                distances.append(distance)

                            # Select the return with the minimum predicted distance
                            min_distance_idx = distances.index(min(distances))
                            predicted_trip = rentals[rental][min_distance_idx]
                            result.append(
                                (
                                    predicted_trip["vehicle_id_rental"],
                                    predicted_trip["timestamp_rental"] * 1000000,
                                    predicted_trip["timestamp_return"] * 1000000,
                                    None,
                                    None,
                                    None,
                                    None,
                                    self.network_mappings[network_name],
                                    self.network_mappings[network_name],
                                    predicted_trip["location_id_rental"],
                                    predicted_trip["location_id_return"],
                                    predicted_trip["battery_rental"],
                                    predicted_trip["battery_return"],
                                    predicted_trip["range_rental"],
                                    predicted_trip["range_return"],
                                    20,
                                    predicted_trip["date"],
                                )
                            )
                    self.logger.info(
                        f"Generated {len(result)} trips for {network_name}"
                    )
            else:
                # Build station-related fields conditionally
                station_fields = ""
                station_joins = ""
                station_selects = ""

                if self.stations:
                    station_fields = """,
                            COALESCE(v.station_id, NULL) AS station_id,
                            COALESCE(sn.station_name_id, NULL) AS station_name_id"""
                    station_joins = """
                        LEFT JOIN station_names sn ON v.station_name = sn.station_name"""
                    station_selects = """
                            LAG(station_id) OVER (PARTITION BY vehicle_id ORDER BY timestamp)::TEXT AS station_id_lend,
                            station_id::TEXT AS station_id_return,
                            LAG(station_name_id) OVER (PARTITION BY vehicle_id ORDER BY timestamp) AS station_name_id_lend,
                            station_name_id AS station_name_id_return,"""
                else:
                    station_fields = ""
                    station_selects = """
                            NULL::TEXT AS station_id_lend,
                            NULL::TEXT AS station_id_return,
                            NULL::INTEGER AS station_name_id_lend,
                            NULL::INTEGER AS station_name_id_return,"""

                vehicle_select_prefix = ""
                if self.gbfs_version == "2":
                    vehicle_select_prefix = "v.bike_id as"

                query = f"""               
                WITH filtered_vehicles AS (
                    SELECT DISTINCT
                        {vehicle_select_prefix} v.vehicle_id::TEXT as vehicle_id,
                        v.last_updated AS timestamp,
                        nm.network_id,
                        lc.location_id,
                        (v.current_fuel_percent * 100)::SMALLINT AS pedelec_battery,
                        v.current_range_meters::INTEGER AS current_range_meters{station_fields}
                    FROM vehicles v
                    LEFT JOIN location_coordinates lc ON (v.lat = lc.lat AND v.lon = lc.lng)
                    LEFT JOIN network_id_mappings nm ON v.network_name = nm.network_name{station_joins}
                ),
                trips AS (
                    SELECT
                    *,
                        vehicle_id,
                        timestamp AS timestamp_return,{station_selects}
                        network_id AS network_name_return,
                        location_id AS location_id_return,
                        pedelec_battery AS pedelec_battery_return,
                        current_range_meters AS current_range_meters_return
                    FROM (
                        SELECT
                            *,
                            LAG(location_id) OVER (PARTITION BY vehicle_id ORDER BY timestamp) AS prev_location_id,
                            LAG(timestamp) OVER (PARTITION BY vehicle_id ORDER BY timestamp) AS prev_timestamp, 
                            LAG(timestamp) OVER (PARTITION BY vehicle_id ORDER BY timestamp) AS timestamp_lend,
                            {"LAG(station_id) OVER (PARTITION BY vehicle_id ORDER BY timestamp) AS prev_station_id," if self.stations else ""}
                            LAG(network_id) OVER (PARTITION BY vehicle_id ORDER BY timestamp) AS network_name_lend,
                            LAG(location_id) OVER (PARTITION BY vehicle_id ORDER BY timestamp) AS location_id_lend,
                            LAG(pedelec_battery) OVER (PARTITION BY vehicle_id ORDER BY timestamp) AS pedelec_battery_lend,
                            LAG(current_range_meters) OVER (PARTITION BY vehicle_id ORDER BY timestamp) AS current_range_meters_lend
                        FROM filtered_vehicles
                    )
                    WHERE
                        prev_location_id IS NOT NULL
                        AND prev_location_id != location_id
                        AND (pedelec_battery_lend IS NULL OR pedelec_battery_return IS NULL OR pedelec_battery_lend != pedelec_battery_return)
                        AND (current_range_meters_lend IS NULL OR current_range_meters_return IS NULL OR current_range_meters_lend != current_range_meters_return)
                        {"AND prev_station_id != station_id" if self.stations else ""}
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
                    CAST(location_id_lend AS INTEGER) AS location_id_lend,
                    CAST(location_id_return AS INTEGER) AS location_id_return,
                    pedelec_battery_lend,
                    pedelec_battery_return,
                    current_range_meters_lend,
                    current_range_meters_return,
                    DATE(timestamp_lend) AS trip_date
                FROM trips
                ORDER BY vehicle_id, timestamp_lend;
                """
                result_query = self.db_connection.execute(query)
                result = result_query.fetch_arrow_table()

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
                if date_key is None:
                    continue

                # Filter rows for this date
                mask = pa.compute.equal(trip_date_col, date_key)
                daily_table = result.filter(mask)

                # Drop the trip_date column and add predicted column
                trips_table = daily_table.select(
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
                    ]
                )

                # Add predicted column
                predicted_col = pa.array(
                    [self.rotating] * trips_table.num_rows, type=pa.bool_()
                )
                trips_table = trips_table.append_column("predicted", predicted_col)

                # Sort by vehicle_id and timestamp_lend
                trips_table = trips_table.sort_by(
                    [("vehicle_id", "ascending"), ("timestamp_lend", "ascending")]
                )

                # Save to parquet file named by date
                filename = f"{date_key}.parquet"
                file_path = export_dir / filename

                if file_path.exists():
                    # Read existing data and append
                    existing_data = pq.read_table(file_path)
                    combined_data = pa.concat_tables([existing_data, trips_table])
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
                        f"Added {trips_table.num_rows} trip records to existing file {filename}"
                    )
                else:
                    pq.write_table(trips_table, file_path, compression="BROTLI")
                    self.logger.info(
                        f"Created {filename} with {trips_table.num_rows} trip records"
                    )

            self.logger.info(f"Trip generation completed. Files saved to {export_dir}")

        except Exception as e:
            self.logger.error(f"Error generating trips: {e}")

    def demand_generator(self):
        """Generate rent and return demand from trip files and save to daily parquet files"""
        try:
            self.logger.info("Starting demand generation...")
            if self.rotating:
                station_fields = ""
                station_joins = ""

                if self.stations:
                    station_fields = """
                            sn.station_name_id AS station_name_id,
                            sn.station_id AS station_id,"""
                    station_joins = """
                        LEFT JOIN station_names sn 
                            ON v.station_name = sn.station_name"""
                else:
                    station_fields = """
                            NULL::INTEGER AS station_name_id,
                            NULL::INTEGER AS station_id,"""

                query = f"""
                WITH first_use AS (
                    SELECT DISTINCT ON (v.{self.id_text})
                        v.{self.id_text} as vehicle_id,
                        lc.location_id, 
                        v.last_updated AS timestamp,{station_fields}
                        nm.network_id AS network_name
                    FROM vehicles v
                    LEFT JOIN network_id_mappings nm 
                        ON v.network_name = nm.network_name
                    LEFT JOIN location_coordinates lc 
                        ON (v.lon = lc.lng 
                        AND v.lat = lc.lat){station_joins}
                    ORDER BY v.{self.id_text}, v.last_updated ASC
                ),
                last_use AS (
                    SELECT DISTINCT ON (v.{self.id_text})
                        v.{self.id_text} as vehicle_id, 
                        lc.location_id, 
                        v.last_updated AS timestamp,{station_fields}
                        nm.network_id AS network_name
                    FROM vehicles v
                    LEFT JOIN network_id_mappings nm 
                        ON v.network_name = nm.network_name
                    LEFT JOIN location_coordinates lc 
                        ON (v.lon = lc.lng 
                        AND v.lat = lc.lat){station_joins}
                    ORDER BY v.{self.id_text}, v.last_updated DESC
                ),
                global_min AS (
                    SELECT MIN(last_updated) AS min_ts FROM vehicles
                ),
                global_max AS (
                    SELECT MAX(last_updated) AS max_ts FROM vehicles
                ),
                return_counts AS (
                    SELECT 
                        f.timestamp,
                        f.location_id,
                        f.station_name_id,
                        f.station_id,
                        f.network_name,
                        COUNT(*) AS return_demand
                    FROM first_use f, global_min g
                    WHERE f.timestamp > g.min_ts   -- ignore very first timestamp
                    GROUP BY f.timestamp, f.location_id, f.station_name_id, f.station_id, f.network_name
                ),
                lend_counts AS (
                    SELECT 
                        l.timestamp,
                        l.location_id,
                        COUNT(*) AS lend_demand,
                        l.station_name_id,
                        l.station_id,
                        l.network_name
                    FROM last_use l, global_max g
                    WHERE l.timestamp < g.max_ts   -- ignore very last timestamp
                    GROUP BY l.timestamp, l.location_id, l.station_name_id, l.station_id, l.network_name
                )
                SELECT DISTINCT
                    COALESCE(r.location_id, l.location_id) AS location_id,
                    COALESCE(r.timestamp, l.timestamp) AS timestamp,
                    COALESCE(r.network_name, l.network_name) AS network_name,
                    COALESCE(r.station_name_id, l.station_name_id) AS station_name_id,
                    COALESCE(r.station_id, l.station_id) AS station_id,
                    COALESCE(l.lend_demand, 0) AS lend_demand,
                    COALESCE(r.return_demand, 0) AS return_demand,
                    DATE(COALESCE(r.timestamp, l.timestamp)) AS demand_date
                FROM return_counts r
                FULL OUTER JOIN lend_counts l 
                    ON r.timestamp = l.timestamp 
                    AND r.location_id = l.location_id
                ORDER BY location_id, timestamp;
                """
            else:
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
                        station_name_id_lend AS station_name_id,
                        station_id_lend AS station_id,
                        COUNT(*)::SMALLINT AS n_lends,
                        0::SMALLINT AS n_returns
                    FROM all_trips
                    WHERE location_id_lend IS NOT NULL AND timestamp_lend IS NOT NULL
                    GROUP BY location_id_lend, timestamp_lend, network_name_lend, station_name_id_lend, station_id_lend
                ),
                return_demand AS (
                    SELECT 
                        location_id_return AS location_id,
                        timestamp_return AS timestamp,
                        network_name_return AS network_name,
                        station_name_id_return AS station_name_id,
                        station_id_return AS station_id,
                        0::SMALLINT AS n_lends,
                        COUNT(*)::SMALLINT AS n_returns
                    FROM all_trips
                    WHERE location_id_return IS NOT NULL AND timestamp_return IS NOT NULL
                    GROUP BY location_id_return, timestamp_return, network_name_return, station_name_id_return, station_id_return
                ),
                combined_demand AS (
                    SELECT * FROM rent_demand
                    UNION ALL
                    SELECT * FROM return_demand
                )
                SELECT DISTINCT
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
            result_query = self.db_connection.execute(query)
            result = result_query.fetch_arrow_table()
            if result.num_rows == 0:
                self.logger.warning("No demand data generated")
                return

            self.logger.info(f"Generated {result.num_rows} demand records")

            # Find lowest and highest timestamps and filter them out
            timestamp_col = result.column("timestamp")
            min_timestamp = pa.compute.min(timestamp_col).as_py()
            max_timestamp = pa.compute.max(timestamp_col).as_py()

            # Filter out min and max timestamps
            mask = pa.compute.and_(
                pa.compute.not_equal(timestamp_col, min_timestamp),
                pa.compute.not_equal(timestamp_col, max_timestamp),
            )
            result = result.filter(mask)

            # Group by demand_date
            demand_date_col = result.column("demand_date")
            unique_dates = pa.compute.unique(demand_date_col).to_pylist()

            # Save to files directly
            export_dir = Path(self.export_data_dir_path) / "demand"
            export_dir.mkdir(parents=True, exist_ok=True)

            for date_key in unique_dates:
                if date_key is None:
                    continue

                # Filter rows for this date
                mask = pa.compute.equal(demand_date_col, date_key)
                daily_table = result.filter(mask)

                # Drop the demand_date column
                demand_table = daily_table.select(
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

                # Sort by location_id and timestamp
                demand_table = demand_table.sort_by(
                    [("location_id", "ascending"), ("timestamp", "ascending")]
                )

                # Save to parquet file named by date
                filename = f"{date_key}.parquet"
                file_path = export_dir / filename

                if file_path.exists():
                    # Read existing data and append
                    existing_data = pq.read_table(file_path)
                    combined_data = pa.concat_tables([existing_data, demand_table])
                    # sort by location_id and timestamp
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
                        f"Added {demand_table.num_rows} demand records to existing file {filename}"
                    )
                else:
                    # Create new file
                    pq.write_table(demand_table, file_path, compression="BROTLI")
                    self.logger.info(
                        f"Created {filename} with {demand_table.num_rows} demand records"
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
            # Build station-related fields conditionally
            station_fields = ""

            if self.stations:
                station_fields = """,
                        v.station_name,
                        v.station_id"""

            query = f"""
                WITH filtered_vehicles AS (
                    SELECT DISTINCT
                        v.{self.id_text} AS vehicle_id, 
                        v.last_updated AS timestamp,
                        v.lat AS lat,
                        v.lon AS lon,
                        v.network_name{station_fields}
                    FROM vehicles v
                ),
                grouped_vehicles AS (
                    SELECT
                        fv.lat,
                        fv.lon,
                        fv.timestamp,
                        fv.network_name,
                        {'fv.station_name,' if self.stations else ''}
                        {'fv.station_id,' if self.stations else ''}
                        COUNT(*)::SMALLINT AS n_vehicles,
                        DATE(fv.timestamp) AS availability_date
                    FROM filtered_vehicles fv
                    GROUP BY fv.lat, fv.lon, fv.timestamp, fv.network_name{', fv.station_name, fv.station_id' if self.stations else ''}
                )
                SELECT DISTINCT
                    lc.location_id::INTEGER AS location_id,
                    gv.timestamp AS timestamp,
                    nm.network_id AS network_name,
                    {'sn.station_name_id' if self.stations else 'NULL::INTEGER'} AS station_name_id,
                    {'gv.station_id::TEXT' if self.stations else 'NULL::TEXT'} AS station_id,
                    gv.n_vehicles,
                    gv.availability_date
                FROM grouped_vehicles gv
                LEFT JOIN location_coordinates lc ON (gv.lat = lc.lat AND gv.lon = lc.lng)
                LEFT JOIN network_id_mappings nm ON gv.network_name = nm.network_name
                {'LEFT JOIN station_names sn ON gv.station_name = sn.station_name' if self.stations else ''}
                ORDER BY gv.timestamp, lc.location_id;
                """

            self.logger.info("Executing availability generation query...")
            result_query = self.db_connection.execute(query)
            result = result_query.fetch_arrow_table()
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
                if date_key is None:
                    continue

                # Filter rows for this date
                mask = pa.compute.equal(availability_date_col, date_key)
                daily_table = result.filter(mask)

                # Drop the availability_date column
                availability_table = daily_table.select(
                    [
                        "location_id",
                        "timestamp",
                        "network_name",
                        "station_name_id",
                        "station_id",
                        "n_vehicles",
                    ]
                )

                # Sort by location_id and timestamp
                availability_table = availability_table.sort_by(
                    [("location_id", "ascending"), ("timestamp", "ascending")]
                )

                # Save to parquet file named by date
                filename = f"{date_key}.parquet"
                file_path = export_dir / filename

                if file_path.exists():
                    # Read existing data and append
                    existing_data = pq.read_table(file_path)
                    combined_data = pa.concat_tables(
                        [existing_data, availability_table]
                    )
                    # sort by location_id and timestamp
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
                        f"Added {availability_table.num_rows} availability records to existing file {filename}"
                    )
                else:
                    # Create new file
                    pq.write_table(availability_table, file_path, compression="BROTLI")
                    self.logger.info(
                        f"Created {filename} with {availability_table.num_rows} availability records"
                    )

            self.logger.info(
                f"Availability generation completed. Files saved to {export_dir}"
            )

        except Exception as e:
            self.logger.error(f"Error generating availability: {e}")

    def getGBFSVersion(self) -> str:
        """Get the GBFS version used by the operator."""
        return self.gbfs_version

    def getProcessorClass(self) -> str:
        """Get the processor class used by the operator."""
        return self.processor_class

    def getProcessingSteps(self) -> list:
        """Get the processing steps configured for the operator."""
        return self.processing_steps

    def getRotating(self) -> bool:
        """Get whether the operator is in rotating mode."""
        return self.rotating
