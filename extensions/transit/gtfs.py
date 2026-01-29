import os
from pathlib import Path
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime, timedelta
import time
import zipfile
import tempfile
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from utils import DataPipelineLogger


class GTFS:
    def __init__(
        self,
        extension_data_dir_path,
        meta_data_dir_path,
        logs_data_dir_path,
        input_data_dir_path,
        num_processes,
        cities,
    ):
        self.extension_data_dir_path = Path(extension_data_dir_path) / "gtfs"
        self.meta_data_dir_path = Path(meta_data_dir_path)
        self.log_file = Path(logs_data_dir_path) / "logs.log"
        self.input_dir = Path(input_data_dir_path)
        self.cities = cities if cities else []
        self.num_processes = num_processes if num_processes and num_processes > 0 else 4

        # Setup logger
        self.logger = DataPipelineLogger.get_logger(
            name=self.__class__.__name__, log_file_path=self.log_file
        )

        # Load shapefiles for filtering
        geodata_path = Path("./data/internal/geodata")
        self.FEDERAL_STATES = gpd.read_file(
            geodata_path / "federal_states/federal_states.shp"
        )
        self.POSTAL_CODES = gpd.read_file(
            geodata_path
            / "postal_codes_with_federal_states/postal_codes_with_federal_states.shp"
        )

        # Optimize shapefiles
        self._optimize_shapefiles()

        # Ensure output directory exists
        os.makedirs(self.extension_data_dir_path, exist_ok=True)

    def _optimize_shapefiles(self):
        """Optimize shapefiles for efficient spatial queries."""
        try:
            # Convert to WGS84 if needed
            if self.FEDERAL_STATES.crs is None:
                self.FEDERAL_STATES = self.FEDERAL_STATES.set_crs("EPSG:4326")
            elif self.FEDERAL_STATES.crs != "EPSG:4326":
                self.FEDERAL_STATES = self.FEDERAL_STATES.to_crs("EPSG:4326")

            if self.POSTAL_CODES.crs is None:
                self.POSTAL_CODES = self.POSTAL_CODES.set_crs("EPSG:4326")
            elif self.POSTAL_CODES.crs != "EPSG:4326":
                self.POSTAL_CODES = self.POSTAL_CODES.to_crs("EPSG:4326")

            # Create spatial indices
            self.FEDERAL_STATES.sindex
            self.POSTAL_CODES.sindex

            self.logger.info("Shapefiles optimized successfully")

        except Exception as e:
            self.logger.error(f"Error optimizing shapefiles: {e}")
            raise

    def run(self):
        """Main entry point for GTFS processing."""
        self.logger.info("Starting GTFS processing...")

        # Get all GTFS ZIP files
        gtfs_zip_files = [
            f
            for f in os.listdir(self.input_dir)
            if f.endswith(".zip") and os.path.isfile(self.input_dir / f)
        ]

        if not gtfs_zip_files:
            self.logger.warning("No GTFS ZIP files found in input directory")
            return

        # Initialize DuckDB connection for final processing
        main_conn = duckdb.connect()
        output_file = self.extension_data_dir_path / "gtfs.parquet"

        try:
            # Create main GTFS table with proper types
            main_conn.execute(
                """
                CREATE TABLE gtfs (
                    route_long_name VARCHAR,
                    route_short_name VARCHAR,
                    route_type INTEGER,
                    location_id INTEGER,
                    arrival_time TIMESTAMP,
                    departure_time TIMESTAMP
                )
            """
            )

            # Process each ZIP file and insert data
            for zip_file in gtfs_zip_files:
                self.logger.info(f"Processing GTFS ZIP file: {zip_file}")
                folder_data = self._process_gtfs_zip(zip_file)

                if folder_data:
                    self.logger.info(
                        f"Inserting {len(folder_data)} records from {zip_file}"
                    )

                    schema = pa.schema(
                        [
                            pa.field("route_long_name", pa.string()),
                            pa.field("route_short_name", pa.string()),
                            pa.field("route_type", pa.int32()),
                            pa.field("location_id", pa.int32()),
                            pa.field(
                                "arrival_time", pa.timestamp("s", tz="Europe/Berlin")
                            ),
                            pa.field(
                                "departure_time", pa.timestamp("s", tz="Europe/Berlin")
                            ),
                        ]
                    )

                    route_long_names = [row["route_long_name"] for row in folder_data]
                    route_short_names = [row["route_short_name"] for row in folder_data]
                    route_types = [row["route_type"] for row in folder_data]
                    location_ids = [row["location_id"] for row in folder_data]
                    arrival_times = [row["arrival_time"] for row in folder_data]
                    departure_times = [row["departure_time"] for row in folder_data]

                    # Convert to PyArrow table for efficient insertion with proper types
                    table = pa.table(
                        {
                            "route_long_name": route_long_names,
                            "route_short_name": route_short_names,
                            "route_type": route_types,
                            "location_id": location_ids,
                            "arrival_time": arrival_times,
                            "departure_time": departure_times,
                        },
                        schema=schema,
                    )

                    table = table.sort_by(
                        [
                            ("arrival_time", "ascending"),
                            ("route_short_name", "ascending"),
                        ]
                    )

                    pq.write_table(
                        table,
                        self.extension_data_dir_path
                        / zip_file.replace(".zip", ".parquet"),
                        compression="BROTLI",
                    )
                else:
                    self.logger.warning("No GTFS data processed")

        except Exception as e:
            self.logger.error(f"Error in main GTFS processing: {e}")
            raise
        finally:
            main_conn.close()

        self.logger.info("GTFS processing completed")

    def _optimize_shapefiles(self):
        """Optimize shapefiles for efficient spatial queries."""
        try:
            # Convert to WGS84 if needed
            if self.FEDERAL_STATES.crs is None:
                self.FEDERAL_STATES = self.FEDERAL_STATES.set_crs("EPSG:4326")
            elif self.FEDERAL_STATES.crs != "EPSG:4326":
                self.FEDERAL_STATES = self.FEDERAL_STATES.to_crs("EPSG:4326")

            if self.POSTAL_CODES.crs is None:
                self.POSTAL_CODES = self.POSTAL_CODES.set_crs("EPSG:4326")
            elif self.POSTAL_CODES.crs != "EPSG:4326":
                self.POSTAL_CODES = self.POSTAL_CODES.to_crs("EPSG:4326")

            # Create spatial indices
            self.FEDERAL_STATES.sindex
            self.POSTAL_CODES.sindex

            self.logger.info("Shapefiles optimized successfully")

        except Exception as e:
            self.logger.error(f"Error optimizing shapefiles: {e}")
            raise

    def _process_gtfs_zip(self, zip_filename):
        """Extract and process a GTFS ZIP file."""
        zip_path = self.input_dir / zip_filename

        # Create temporary directory for extraction
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            try:
                # Extract ZIP file
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(temp_path)
                    self.logger.info(f"Extracted {zip_filename} to temporary directory")

                # Process the extracted files
                return self._process_gtfs_folder_from_path(temp_path, zip_filename)

            except zipfile.BadZipFile:
                self.logger.error(f"Invalid ZIP file: {zip_filename}")
                return []
            except Exception as e:
                self.logger.error(f"Error processing ZIP file {zip_filename}: {e}")
                return []
            # Temporary directory is automatically cleaned up when exiting the context

    def _process_gtfs_folder_from_path(self, folder_path, source_name):
        """Process GTFS files from a given folder path."""
        # Check required files exist
        required_files = [
            "stops.txt",
            "stop_times.txt",
            "trips.txt",
            "routes.txt",
            "calendar.txt",
        ]
        for file in required_files:
            if not (folder_path / file).exists():
                self.logger.warning(f"Missing required file {file} in {source_name}")
                return []

        # Initialize DuckDB connection
        conn = duckdb.connect()
        conn.execute("INSTALL spatial;")
        conn.execute("LOAD spatial;")

        try:
            # Load GTFS files into DuckDB
            self._load_gtfs_files(conn, folder_path)

            # Filter stops by geographic location
            filtered_stops = self._filter_stops_by_location(conn)

            if not filtered_stops:
                self.logger.warning(
                    f"No stops found in target regions for {source_name}"
                )
                return []

            # Get location coordinates and create location mapping
            location_mapping = self._create_location_mapping(filtered_stops)

            # Process GTFS data
            gtfs_data = self._process_gtfs_data(conn, location_mapping, source_name)

            return gtfs_data

        except Exception as e:
            self.logger.error(f"Error processing GTFS folder {source_name}: {e}")
            return []
        finally:
            conn.close()

    def _load_gtfs_files(self, conn, folder_path):
        """Load GTFS files into DuckDB tables with robust CSV parsing."""
        # CSV options for handling malformed GTFS files
        csv_options = """
            ignore_errors=true,
            null_padding=true,
            quote='"',
            escape='"',
            auto_detect=true,
            header=true
        """

        try:
            # Load stops
            conn.execute(
                f"""
                CREATE TABLE stops AS 
                SELECT * FROM read_csv('{folder_path / "stops.txt"}', {csv_options})
            """
            )

            # Load stop_times
            conn.execute(
                f"""
                CREATE TABLE stop_times AS 
                SELECT * FROM read_csv('{folder_path / "stop_times.txt"}', {csv_options})
            """
            )

            # Load trips
            conn.execute(
                f"""
                CREATE TABLE trips AS 
                SELECT * FROM read_csv('{folder_path / "trips.txt"}', {csv_options})
            """
            )

            # Load routes
            conn.execute(
                f"""
                CREATE TABLE routes AS 
                SELECT * FROM read_csv('{folder_path / "routes.txt"}', {csv_options})
            """
            )

            # Load calendar
            conn.execute(
                f"""
                CREATE TABLE calendar AS 
                SELECT * FROM read_csv('{folder_path / "calendar.txt"}', {csv_options})
            """
            )

            self.logger.info("Successfully loaded all GTFS files")

        except Exception as e:
            self.logger.error(f"Error loading GTFS files: {e}")
            # Try with read_csv_auto as fallback
            self.logger.info("Retrying with read_csv_auto...")

            try:
                # Load stops
                conn.execute(
                    f"""
                    CREATE TABLE stops AS 
                    SELECT * FROM read_csv_auto('{folder_path / "stops.txt"}', 
                        ignore_errors=true, null_padding=true)
                """
                )

                # Load stop_times
                conn.execute(
                    f"""
                    CREATE TABLE stop_times AS 
                    SELECT * FROM read_csv_auto('{folder_path / "stop_times.txt"}', 
                        ignore_errors=true, null_padding=true)
                """
                )

                # Load trips
                conn.execute(
                    f"""
                    CREATE TABLE trips AS 
                    SELECT * FROM read_csv_auto('{folder_path / "trips.txt"}', 
                        ignore_errors=true, null_padding=true)
                """
                )

                # Load routes
                conn.execute(
                    f"""
                    CREATE TABLE routes AS 
                    SELECT * FROM read_csv_auto('{folder_path / "routes.txt"}', 
                        ignore_errors=true, null_padding=true)
                """
                )

                # Load calendar
                conn.execute(
                    f"""
                    CREATE TABLE calendar AS 
                    SELECT * FROM read_csv_auto('{folder_path / "calendar.txt"}', 
                        ignore_errors=true, null_padding=true)
                """
                )

                self.logger.info("Successfully loaded GTFS files with read_csv_auto")

            except Exception as e2:
                self.logger.error(
                    f"Failed to load GTFS files even with read_csv_auto: {e2}"
                )
                raise

    def _filter_stops_by_location(self, conn):
        """Filter stops by geographic location using spatial queries."""
        # Get stops with valid coordinates - handle empty strings and non-numeric values
        stops_result = conn.execute(
            """
            SELECT stop_id, stop_name, 
                   CAST(stop_lat AS DOUBLE) as lat, 
                   CAST(stop_lon AS DOUBLE) as lng
            FROM stops 
            WHERE stop_lat IS NOT NULL 
            AND stop_lon IS NOT NULL
            AND TRIM(CAST(stop_lat AS VARCHAR)) != ''
            AND TRIM(CAST(stop_lon AS VARCHAR)) != ''
            AND TRY_CAST(stop_lat AS DOUBLE) IS NOT NULL
            AND TRY_CAST(stop_lon AS DOUBLE) IS NOT NULL
            AND TRY_CAST(stop_lat AS DOUBLE) BETWEEN -90 AND 90
            AND TRY_CAST(stop_lon AS DOUBLE) BETWEEN -180 AND 180
        """
        ).fetchall()

        filtered_stops = []
        lock = threading.Lock()

        def process_stop(stop_data):
            """Process a single stop for geographic filtering."""
            stop_id, stop_name, lat, lng = stop_data
            try:
                lat, lng = float(lat), float(lng)

                # Check if point is in target regions using spatial lookup
                if self._is_point_in_target_region(lat, lng):
                    return {
                        "stop_id": stop_id,
                        "stop_name": stop_name,
                        "lat": lat,
                        "lng": lng,
                    }
            except (ValueError, TypeError):
                pass
            return None

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.num_processes) as executor:
            # Submit all tasks
            futures = [
                executor.submit(process_stop, stop_data) for stop_data in stops_result
            ]

            # Process completed tasks with progress bar
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Filtering stops by location",
            ):
                result = future.result()
                if result is not None:
                    with lock:
                        filtered_stops.append(result)

        self.logger.info(
            f"Filtered {len(filtered_stops)} stops from {len(stops_result)} total stops"
        )
        return filtered_stops

    def _is_point_in_target_region(self, lat, lng):
        """Check if a point is in target regions (Germany or specified cities)."""
        try:
            point = Point(lng, lat)

            # Check if in any German federal state
            possible_matches_index = list(
                self.FEDERAL_STATES.sindex.intersection(point.bounds)
            )
            possible_matches = self.FEDERAL_STATES.iloc[possible_matches_index]
            precise_matches = possible_matches[possible_matches.contains(point)]

            if not precise_matches.empty:
                # If specific cities are specified, check postal codes
                if self.cities:
                    postal_possible = list(
                        self.POSTAL_CODES.sindex.intersection(point.bounds)
                    )
                    postal_matches = self.POSTAL_CODES.iloc[postal_possible]
                    postal_precise = postal_matches[postal_matches.contains(point)]

                    if not postal_precise.empty:
                        city_name = postal_precise.iloc[0].get("city", "").lower()
                        return any(city.lower() in city_name for city in self.cities)
                    return False
                else:
                    return True

            return False

        except Exception as e:
            self.logger.warning(f"Error checking point {lat}, {lng}: {e}")
            return False

    def _create_location_mapping(self, filtered_stops):
        """Create mapping from stop coordinates to location IDs."""
        # Load existing location coordinates if available
        coordinates_file = self.meta_data_dir_path / "location_coordinates.parquet"
        location_mapping = {}

        # Get unique coordinates from filtered stops
        unique_coords = set()
        for stop in filtered_stops:
            lat = round(float(stop["lat"]), 3)
            lng = round(float(stop["lng"]), 3)
            unique_coords.add((lat, lng))

        if not unique_coords:
            self.logger.warning("No coordinates found in filtered stops")
            return location_mapping

        # Convert to list for processing
        new_coords = list(unique_coords)

        if coordinates_file.exists():
            # Load existing coordinates
            existing_table = pq.read_table(coordinates_file)
            existing_coords = set(
                zip(
                    existing_table["lat"].to_pylist(),
                    existing_table["lng"].to_pylist(),
                )
            )

            # Get existing location_ids and create mapping
            location_ids = existing_table["location_id"].to_pylist()
            lats = existing_table["lat"].to_pylist()
            lngs = existing_table["lng"].to_pylist()

            coord_to_id = {}
            for location_id, lat, lng in zip(location_ids, lats, lngs):
                coord_to_id[(round(lat, 3), round(lng, 3))] = location_id

            max_location_id = max(location_ids)

            # Filter out coordinates that already exist
            new_coords = [
                (lat, lng)
                for lat, lng in new_coords
                if (lat, lng) not in existing_coords
            ]

            # Add new coordinates if any
            if new_coords:
                self.logger.info(
                    f"Adding {len(new_coords)} new coordinates to existing file"
                )

                new_lats, new_lngs = zip(*new_coords)
                new_location_ids = [
                    max_location_id + i + 1 for i in range(len(new_coords))
                ]

                # Add new coordinates to mapping
                for (lat, lng), location_id in zip(new_coords, new_location_ids):
                    coord_to_id[(lat, lng)] = location_id

                # Create new data table
                new_data = pa.table(
                    {"location_id": new_location_ids, "lat": new_lats, "lng": new_lngs}
                )

                # Append to existing file
                existing_data = pq.read_table(coordinates_file)
                combined_data = pa.concat_tables([existing_data, new_data])

                pq.write_table(combined_data, coordinates_file, compression="BROTLI")
            else:
                self.logger.info("No new coordinates to add")

        else:
            # Create new coordinates file
            max_location_id = 0
            self.logger.info(
                f"Creating new coordinates file with {len(new_coords)} coordinates"
            )

            new_lats, new_lngs = zip(*new_coords)
            new_location_ids = [max_location_id + i + 1 for i in range(len(new_coords))]

            coord_to_id = {}
            for (lat, lng), location_id in zip(new_coords, new_location_ids):
                coord_to_id[(lat, lng)] = location_id

            new_table = pa.table(
                {"location_id": new_location_ids, "lat": new_lats, "lng": new_lngs}
            )
            pq.write_table(new_table, coordinates_file, compression="BROTLI")

        # Map stop coordinates to location IDs
        for stop in filtered_stops:
            rounded_key = (round(stop["lat"], 3), round(stop["lng"], 3))
            if rounded_key in coord_to_id:
                location_mapping[stop["stop_id"]] = coord_to_id[rounded_key]

        self.logger.info(f"Created location mapping for {len(location_mapping)} stops")
        return location_mapping

    def _process_gtfs_data(self, conn, location_mapping, folder_name):
        """Process GTFS data and create final output."""
        # Create stop filter for SQL
        valid_stop_ids = list(location_mapping.keys())
        # Convert all stop_ids to strings to handle mixed int/string types
        stop_ids_str = (
            "'" + "','".join(str(stop_id) for stop_id in valid_stop_ids) + "'"
        )

        # Get merged GTFS data using SQL
        gtfs_result = conn.execute(
            f"""
            SELECT 
                r.route_long_name,
                r.route_short_name,
                r.route_type,
                st.stop_id,
                st.arrival_time,
                st.departure_time,
                c.monday, c.tuesday, c.wednesday, c.thursday, c.friday, c.saturday, c.sunday,
                c.start_date,
                c.end_date
            FROM stop_times st
            JOIN trips t ON st.trip_id = t.trip_id
            JOIN routes r ON t.route_id = r.route_id
            JOIN calendar c ON t.service_id = c.service_id
            WHERE CAST(st.stop_id AS VARCHAR) IN ({stop_ids_str})
            AND st.arrival_time IS NOT NULL
            AND st.departure_time IS NOT NULL
            AND TRIM(CAST(st.arrival_time AS VARCHAR)) != ''
            AND TRIM(CAST(st.departure_time AS VARCHAR)) != ''
            AND LENGTH(TRIM(CAST(st.arrival_time AS VARCHAR))) >= 8
            AND LENGTH(TRIM(CAST(st.departure_time AS VARCHAR))) >= 8
            AND CAST(st.arrival_time AS VARCHAR) LIKE '%:%:%'
            AND CAST(st.departure_time AS VARCHAR) LIKE '%:%:%'
        """
        ).fetchall()

        gtfs_data = []

        for row in gtfs_result:
            (
                route_long_name,
                route_short_name,
                route_type,
                stop_id,
                arrival_time,
                departure_time,
                monday,
                tuesday,
                wednesday,
                thursday,
                friday,
                saturday,
                sunday,
                start_date,
                end_date,
            ) = row

            try:
                # Parse dates
                start_dt = datetime.strptime(str(start_date), "%Y%m%d")
                end_dt = datetime.strptime(str(end_date), "%Y%m%d")

                # Parse times                
                arrival_parts = str(arrival_time).split(":")
                departure_parts = str(departure_time).split(":")

                if len(arrival_parts) != 3 or len(departure_parts) != 3:
                    continue

                arrival_hour = int(arrival_parts[0])
                arrival_minute = int(arrival_parts[1])
                arrival_second = int(arrival_parts[2])

                departure_hour = int(departure_parts[0])
                departure_minute = int(departure_parts[1])
                departure_second = int(departure_parts[2])

                # Handle times >= 24:00:00
                arrival_days_offset = arrival_hour // 24
                arrival_hour = arrival_hour % 24
                departure_days_offset = departure_hour // 24
                departure_hour = departure_hour % 24

                # Get days of week that this service runs
                days_active = []
                if monday:
                    days_active.append(0)  # Monday = 0
                if tuesday:
                    days_active.append(1)
                if wednesday:
                    days_active.append(2)
                if thursday:
                    days_active.append(3)
                if friday:
                    days_active.append(4)
                if saturday:
                    days_active.append(5)
                if sunday:
                    days_active.append(6)

                # Generate timestamps for each active day in the date range
                current_date = start_dt
                while current_date <= end_dt:
                    if current_date.weekday() in days_active:
                        # Calculate actual datetime
                        arrival_dt = current_date + timedelta(
                            days=arrival_days_offset,
                            hours=arrival_hour,
                            minutes=arrival_minute,
                            seconds=arrival_second,
                        )
                        departure_dt = current_date + timedelta(
                            days=departure_days_offset,
                            hours=departure_hour,
                            minutes=departure_minute,
                            seconds=departure_second,
                        )

                        arrival_timestamp = int(arrival_dt.timestamp())
                        departure_timestamp = int(departure_dt.timestamp())

                        gtfs_data.append(
                            {
                                "route_long_name": route_long_name or "",
                                "route_short_name": route_short_name or "",
                                "route_type": int(route_type) if route_type else 0,
                                "location_id": location_mapping[stop_id],
                                "arrival_time": arrival_timestamp,
                                "departure_time": departure_timestamp,
                            }
                        )

                    current_date += timedelta(days=1)

            except (ValueError, TypeError, KeyError) as e:
                self.logger.warning(f"Error parsing GTFS record: {e}")
                continue

        self.logger.info(f"Processed {len(gtfs_data)} GTFS records from {folder_name}")
        return gtfs_data
