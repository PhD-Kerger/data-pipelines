import os
import shutil
import requests
import datetime
import pytz
import csv
from geopy.geocoders import Nominatim
import math
from bs4 import BeautifulSoup
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

from utils import DataPipelineLogger


class DWD:
    # Class-level constants to reduce instance variables
    WEATHER_TYPES = ["wind", "solar", "precipitation", "air_temperature"]

    BASE_URL = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes"

    # Configuration for each weather type
    WEATHER_CONFIG = {
        "wind": {
            "url_suffix": "wind",
            "station_file": "zehn_min_ff_Beschreibung_Stationen.txt",
            "data_columns": ["FF_10", "DD_10"],  # wind speed, wind direction
        },
        "solar": {
            "url_suffix": "solar",
            "station_file": "zehn_min_sd_Beschreibung_Stationen.txt",
            "data_columns": ["SD_10"],  # solar duration
        },
        "precipitation": {
            "url_suffix": "precipitation",
            "station_file": "zehn_min_rr_Beschreibung_Stationen.txt",
            "data_columns": ["RWS_10"],  # precipitation
        },
        "air_temperature": {
            "url_suffix": "air_temperature",
            "station_file": "zehn_min_tu_Beschreibung_Stationen.txt",
            "data_columns": ["TM5_10", "RF_10"],  # temperature, humidity
        },
    }

    def __init__(
        self,
        extension_data_dir_path,
        meta_data_dir_path,
        input_data_dir_path,
        logs_data_dir_path,
        start_date,
        end_date,
        cities,
    ):
        self.extension_data_dir_path = extension_data_dir_path + "/weather"
        self.meta_data_dir_path = meta_data_dir_path
        self.log_file = Path(logs_data_dir_path) / "logs.log"
        self.extension_temp_dir_path = extension_data_dir_path + "/tmp"

        # Setup logger
        self.logger = DataPipelineLogger.get_logger(
            name=self.__class__.__name__, log_file_path=self.log_file
        )

        os.makedirs(self.extension_data_dir_path, exist_ok=True)
        os.makedirs(self.extension_temp_dir_path, exist_ok=True)

        self.start_date = start_date
        self.end_date = end_date
        self.cities = cities if cities else []

        # Initialize data structures using class constants
        self.city_coordinates = []
        self.filtered_weather_stations = {wtype: [] for wtype in self.WEATHER_TYPES}
        self.weather_data = {wtype: [] for wtype in self.WEATHER_TYPES}

    @property
    def start_timestamp(self):
        """Convert start_date to timestamp once and cache it"""
        if not hasattr(self, "_start_timestamp"):
            self._start_timestamp = int(
                datetime.datetime.strptime(self.start_date, "%Y%m%d")
                .replace(tzinfo=pytz.UTC)
                .timestamp()
            )
        return self._start_timestamp

    @property
    def end_timestamp(self):
        """Convert end_date to timestamp once and cache it"""
        if not hasattr(self, "_end_timestamp"):
            self._end_timestamp = int(
                datetime.datetime.strptime(self.end_date, "%Y%m%d")
                .replace(tzinfo=pytz.UTC)
                .timestamp()
            )
        return self._end_timestamp

    @property
    def start_datetime(self):
        """Convert start_date to datetime once and cache it"""
        if not hasattr(self, "_start_datetime"):
            self._start_datetime = datetime.datetime.strptime(self.start_date, "%Y%m%d")
        return self._start_datetime

    @property
    def end_datetime(self):
        """Convert end_date to datetime once and cache it"""
        if not hasattr(self, "_end_datetime"):
            self._end_datetime = datetime.datetime.strptime(self.end_date, "%Y%m%d")
        return self._end_datetime

    def run(self):
        """Main method to run the DWD weather data processing"""
        if len(self.cities) == 0:
            self.logger.warning("No cities provided for DWD extension. Terminating.")
            return

        # Step 1: Get city coordinates and store as class variable
        self._get_city_coordinates()

        # Step 2: Get filtered weather stations using the stored coordinates
        self._get_filtered_weather_stations()

        # Step 3: Download and process weather data
        self.download_weather_data()

        # Step 4: Process the downloaded weather data
        self.process_weather_data()

        # Step 5: Export weather data to parquet format
        self.export_weather_data_to_parquet()

        self.logger.info("DWD weather processing completed")

    def _get_filtered_weather_stations(self):
        """Download weather station lists, find nearest stations for cities, and filter them in one operation"""
        self.logger.info("Processing weather stations (download, find nearest, filter)")

        for weather_type, config in self.WEATHER_CONFIG.items():
            self.logger.info(f"Processing {weather_type} stations")

            # Build station URL
            station_url = f"{self.BASE_URL}/{config['url_suffix']}/historical/{config['station_file']}"

            # Download and parse station list
            r = requests.get(station_url, allow_redirects=True, verify=False)
            content = r.content.decode("utf-8", errors="ignore")
            lines = content.split("\r\n")[2:-1]  # Skip header and empty last line

            # Parse fixed-width format
            parsed_stations = []
            for line in lines:
                if len(line.strip()) == 0:
                    continue

                station_data = [
                    line[:5].strip(),  # station_id
                    line[5:14].strip(),  # start_date
                    line[14:23].strip(),  # end_date
                    line[30:39].strip(),  # elevation
                    line[39:50].strip(),  # latitude
                    line[50:60].strip(),  # longitude
                    line[60:99].strip(),  # name
                ]

                # Convert dates to timestamps and validate
                start_date = datetime.datetime.strptime(
                    station_data[1], "%Y%m%d"
                ).replace(tzinfo=pytz.UTC)
                end_date = datetime.datetime.strptime(
                    station_data[2], "%Y%m%d"
                ).replace(tzinfo=pytz.UTC)

                # Only include stations that cover our date range
                if (
                    start_date.timestamp() <= self.start_timestamp
                    and end_date.timestamp() >= self.end_timestamp
                ):
                    parsed_stations.append(
                        [
                            int(station_data[0]),
                            int(start_date.timestamp()),
                            int(end_date.timestamp()),
                            int(station_data[3]),
                            float(station_data[4]),
                            float(station_data[5]),
                            station_data[6],
                        ]
                    )

            if not parsed_stations:
                self.logger.warning(f"No {weather_type} stations found for date range")
                continue

            # Find nearest station for each city
            nearest_stations = {}
            for city in self.city_coordinates:
                city_lat = float(city["latitude"])
                city_lon = float(city["longitude"])

                min_distance = float("inf")
                nearest_station = None

                for station in parsed_stations:
                    distance = self._haversine_distance(
                        city_lat, city_lon, station[4], station[5]
                    )
                    if distance < min_distance:
                        min_distance = distance
                        nearest_station = station

                if nearest_station:
                    nearest_stations[city["name"]] = nearest_station[0]  # station_id

            # Filter stations to only include nearest ones
            self.filtered_weather_stations[weather_type] = [
                station
                for station in parsed_stations
                if station[0] in set(nearest_stations.values())
            ]

        self.logger.info("Weather station processing completed")

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate the haversine distance between two points."""
        # Convert latitude and longitude from degrees to radians
        lat1 = math.radians(lat1)
        lon1 = math.radians(lon1)
        lat2 = math.radians(lat2)
        lon2 = math.radians(lon2)

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        # Radius of Earth in kilometers
        radius_of_earth_km = 6371
        distance = radius_of_earth_km * c

        return distance

    def _get_city_coordinates(self):
        """Get the coordinates of the cities."""
        self.logger.info("Getting city coordinates")
        geolocator = Nominatim(user_agent="weather_station_locator")
        self.city_coordinates = []  # Store as class variable

        for city in self.cities:
            try:
                location = geolocator.geocode(city)
                if location:
                    self.city_coordinates.append(
                        {
                            "name": city,
                            "latitude": round(location.latitude, 4),
                            "longitude": round(location.longitude, 4),
                        }
                    )
                else:
                    self.logger.warning(f"Could not geocode city: {city}")
            except Exception as e:
                self.logger.error(f"Error geocoding {city}: {e}")

    def download_weather_data(self):
        """Download weather data for each weather type"""
        for weather_type, config in self.WEATHER_CONFIG.items():
            base_url = f"{self.BASE_URL}/{config['url_suffix']}/"
            for time_range in ["historical", "recent"]:
                url_filter = base_url + time_range + "/"
                r = requests.get(url_filter, verify=False)
                c = r.content

                # need to parse the html to find the hrefs easily
                soup = BeautifulSoup(c, "html.parser")

                for link in soup.find_all("a"):
                    try:
                        for station in self.filtered_weather_stations[weather_type]:
                            weather_station_id = str(
                                station[0]
                            )  # station_id is first element
                            # id is 5 digits long. if not, add leading zeros
                            weather_station_id = weather_station_id.zfill(5)
                            # check href link if weather_station_id is in it. then we download the file and unzip it and delete the zip file
                            if weather_station_id in link.get("href"):
                                r = requests.get(
                                    url_filter + link.get("href"),
                                    allow_redirects=True,
                                    verify=False,
                                )
                                os.makedirs(
                                    self.extension_temp_dir_path
                                    + "/raw/"
                                    + weather_type,
                                    exist_ok=True,
                                )
                                open(
                                    self.extension_temp_dir_path
                                    + "/raw/"
                                    + weather_type
                                    + "/"
                                    + link.get("href"),
                                    "wb",
                                ).write(r.content)
                                # unzip and delete zip file. only keep txt files. delete all other files
                                os.system(
                                    "unzip "
                                    + str(self.extension_temp_dir_path)
                                    + "/raw/"
                                    + weather_type
                                    + "/"
                                    + link.get("href")
                                    + " -d "
                                    + str(self.extension_temp_dir_path)
                                    + "/raw/"
                                    + weather_type
                                    + "/"
                                )
                                os.system(
                                    "rm "
                                    + str(self.extension_temp_dir_path)
                                    + "/raw/"
                                    + weather_type
                                    + "/"
                                    + link.get("href")
                                )
                    except:
                        pass

    def process_weather_data(self):
        """Process weather data for all weather types from config"""
        for weather_type in self.WEATHER_CONFIG.keys():
            data_dir = self.extension_temp_dir_path + "/raw/" + weather_type

            if not os.path.exists(data_dir):
                self.logger.warning(f"No data directory found for {weather_type}")
                continue

            all_data = []
            config = self.WEATHER_CONFIG[weather_type]

            for file in os.listdir(data_dir):
                if not file.endswith(".txt"):
                    continue

                # Skip files with dates before our start date
                try:
                    file_end_date = file.split("_")[-2]
                    if int(file_end_date) < int(self.start_date):
                        continue
                except (IndexError, ValueError):
                    continue

                file_path = data_dir + "/" + file

                try:
                    with open(
                        file_path, "r", encoding="utf-8", errors="ignore"
                    ) as csvfile:
                        reader = csv.DictReader(csvfile, delimiter=";")

                        for row in reader:
                            try:
                                weather_station_id = row["STATIONS_ID"].strip()
                                mess_datum = datetime.datetime.strptime(
                                    row["MESS_DATUM"], "%Y%m%d%H%M"
                                )

                                # Check if date is in our range
                                if not (
                                    self.start_datetime
                                    <= mess_datum
                                    <= self.end_datetime
                                ):
                                    continue

                                # Find station info
                                station_info = None
                                for station in self.filtered_weather_stations[
                                    weather_type
                                ]:
                                    if str(station[0]) == weather_station_id:
                                        station_info = station
                                        break

                                if not station_info:
                                    continue

                                # Build data row with timestamp
                                data_row = [
                                    int(mess_datum.replace(tzinfo=pytz.UTC).timestamp())
                                ]

                                # Add weather data columns based on configuration
                                for col in config["data_columns"]:
                                    value = row.get(col, "").strip()
                                    data_row.append(value)

                                # Add station coordinates (lat, lon)
                                data_row.extend(
                                    [station_info[4], station_info[5]]
                                )  # lat, lon

                                all_data.append(data_row)

                            except (KeyError, ValueError, IndexError) as e:
                                continue  # Skip invalid rows

                except Exception as e:
                    self.logger.error(f"Error processing file {file}: {e}")
                    continue

            if all_data:
                self.weather_data[weather_type] = all_data
                self.logger.info(
                    f"Processed {len(all_data)} records for {weather_type}"
                )
            else:
                self.logger.warning(f"No valid data found for {weather_type}")

    def _get_unique_location_coordinates(self):
        """Extract unique coordinates and append to location_coordinates.parquet"""
        try:
            # Define the meta data file path first
            meta_dir = Path(self.meta_data_dir_path)
            meta_dir.mkdir(parents=True, exist_ok=True)
            coordinates_file = meta_dir / "location_coordinates.parquet"

            # Get unique coordinates from weather station data
            unique_coords = set()
            for weather_type in self.WEATHER_TYPES:
                for data_row in self.weather_data.get(weather_type, []):
                    if len(data_row) >= 3:  # timestamp + data columns + lat + lon
                        lat = round(float(data_row[-2]), 3)
                        lng = round(float(data_row[-1]), 3)
                        unique_coords.add((lat, lng))

            if not unique_coords:
                self.logger.warning("No coordinates found in weather data")
                return

            # Convert to list for processing
            new_coords = list(unique_coords)

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

        except Exception as e:
            self.logger.error(f"Error extracting coordinates: {e}")

    def _map_coordinates_to_location_ids(self):
        """Map coordinates in weather data to location IDs from the parquet file"""
        try:
            # Read the location coordinates file
            meta_dir = Path(self.meta_data_dir_path)
            coordinates_file = meta_dir / "location_coordinates.parquet"

            if not coordinates_file.exists():
                self.logger.error(
                    "location_coordinates.parquet not found. Run _get_unique_location_coordinates first."
                )
                return {}

            # Read the parquet file and create a mapping
            location_data = pq.read_table(coordinates_file)
            coord_to_id = {}

            location_ids = location_data.column("location_id").to_pylist()
            lats = location_data.column("lat").to_pylist()
            lngs = location_data.column("lng").to_pylist()

            for location_id, lat, lng in zip(location_ids, lats, lngs):
                coord_to_id[(round(lat, 3), round(lng, 3))] = location_id

            return coord_to_id

        except Exception as e:
            self.logger.error(f"Error mapping coordinates to location IDs: {e}")
            return {}

    def export_weather_data_to_parquet(self):
        """Export weather data to parquet file with proper schema"""
        try:
            # First ensure location coordinates are up to date
            self._get_unique_location_coordinates()

            # Get coordinate to location ID mapping
            coord_to_id = self._map_coordinates_to_location_ids()
            if not coord_to_id:
                self.logger.error("No coordinate mappings available")
                return

            # Prepare data for export
            export_data = []

            for weather_type in self.WEATHER_TYPES:
                weather_records = self.weather_data.get(weather_type, [])

                for data_row in weather_records:
                    if (
                        len(data_row) < 3
                    ):  # Need at least timestamp + some data + coordinates
                        continue

                    timestamp = data_row[0]
                    lat = round(float(data_row[-2]), 3)
                    lng = round(float(data_row[-1]), 3)

                    # Get location ID from coordinates
                    location_id = coord_to_id.get((lat, lng))
                    if location_id is None:
                        continue

                    # Initialize weather record with defaults
                    weather_record = {
                        "location_id": location_id,
                        "timestamp": timestamp,
                        "temperature": None,
                        "humidity": None,
                        "precipitation": None,
                        "wind_speed": None,
                        "wind_direction": None,
                    }

                    # Map data based on weather type
                    if weather_type == "air_temperature":
                        # data_row format: [timestamp, TM5_10, RF_10, lat, lon]
                        if len(data_row) >= 4:
                            temp_value = (
                                data_row[1].strip()
                                if data_row[1].strip() != "-999"
                                else None
                            )
                            humidity_value = (
                                data_row[2].strip()
                                if data_row[2].strip() != "-999"
                                else None
                            )
                            weather_record["temperature"] = (
                                float(temp_value) if temp_value else None
                            )
                            weather_record["humidity"] = (
                                float(humidity_value) if humidity_value else None
                            )

                    elif weather_type == "precipitation":
                        # data_row format: [timestamp, RWS_10, lat, lon]
                        if len(data_row) >= 3:
                            precip_value = (
                                data_row[1].strip()
                                if data_row[1].strip() != "-999"
                                else None
                            )
                            weather_record["precipitation"] = (
                                float(precip_value) if precip_value else None
                            )

                    elif weather_type == "wind":
                        # data_row format: [timestamp, FF_10, DD_10, lat, lon]
                        if len(data_row) >= 4:
                            wind_speed_value = (
                                data_row[1].strip()
                                if data_row[1].strip() != "-999"
                                else None
                            )
                            wind_dir_value = (
                                data_row[2].strip()
                                if data_row[2].strip() != "-999"
                                else None
                            )
                            weather_record["wind_speed"] = (
                                float(wind_speed_value) if wind_speed_value else None
                            )
                            weather_record["wind_direction"] = (
                                int(float(wind_dir_value)) if wind_dir_value else None
                            )

                    export_data.append(weather_record)

            if not export_data:
                self.logger.warning("No weather data to export")
                return

            # Group data by location_id and timestamp to merge different weather types
            merged_data = {}
            for record in export_data:
                key = (record["location_id"], record["timestamp"])
                if key not in merged_data:
                    merged_data[key] = record.copy()
                else:
                    # Merge non-None values
                    for field in [
                        "temperature",
                        "humidity",
                        "precipitation",
                        "wind_speed",
                        "wind_direction",
                    ]:
                        if record[field] is not None:
                            merged_data[key][field] = record[field]

            # Convert to lists for PyArrow
            final_records = list(merged_data.values())

            location_ids = [r["location_id"] for r in final_records]
            timestamps = [r["timestamp"] for r in final_records]
            temperatures = [r["temperature"] for r in final_records]
            humidities = [r["humidity"] for r in final_records]
            precipitations = [r["precipitation"] for r in final_records]
            wind_speeds = [r["wind_speed"] for r in final_records]
            wind_directions = [r["wind_direction"] for r in final_records]

            # Create PyArrow table
            weather_table = pa.table(
                {
                    "location_id": pa.array(location_ids, type=pa.int32()),
                    "timestamp": pa.array(timestamps, type=pa.timestamp("s", tz="UTC")),
                    "temperature": pa.array(temperatures, type=pa.float64()),
                    "humidity": pa.array(humidities, type=pa.float64()),
                    "precipitation": pa.array(precipitations, type=pa.float64()),
                    "wind_speed": pa.array(wind_speeds, type=pa.float64()),
                    "wind_direction": pa.array(wind_directions, type=pa.int32()),
                }
            )

            # Write to parquet file
            output_file = Path(self.extension_data_dir_path) / "dwd.parquet"
            # If file exists, delete it first
            if output_file.exists():
                output_file.unlink()

            pq.write_table(weather_table, output_file, compression="BROTLI")

            # delete the tmp folder
            if Path(self.extension_temp_dir_path).exists():
                shutil.rmtree(Path(self.extension_temp_dir_path))

            self.logger.info(
                f"Exported {len(final_records)} weather records to {output_file}"
            )

        except Exception as e:
            self.logger.error(f"Error exporting weather data to parquet: {e}")
