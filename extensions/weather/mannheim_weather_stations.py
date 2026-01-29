import datetime
import os
from pathlib import Path
import shutil
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow_ops as pa_ops
import requests
import json

import pytz

from utils.data_pipeline_logger import DataPipelineLogger


class MannheimWeatherStations:
    """Extension for Mannheim weather stations."""

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

        # start and end date are in format YYYYMMDD. they need to be converted to yyyy-mm-dd for the mannheim api
        self.start_date = start_date
        self.end_date = end_date
        self.station_ids = [287, 288, 292]

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
        """Main method to run the Mannheim weather data processing."""

        self.logger.info(
            "Starting weather data processing for Mannheim specific stations"
        )

        # Step 1: Download and process weather data
        self.download_weather_data()

        # Step 2: Download station metadata
        self.download_station_data()

        # Step 3: Process the downloaded weather data
        self.process_weather_data()

        # # Step 3: Export weather data to parquet format
        self.export_weather_data_to_parquet()

        self.logger.info(
            "Completed weather data processing for Mannheim specific stations"
        )

    def download_station_data(self):
        """Download station metadata for Mannheim stations."""
        self.logger.info("Downloading station metadata for Mannheim stations...")
        url = "https://stadtklimaanalyse-mannheim.de/wp-json/climate-data/v1/station"
        response = requests.get(url)
        station_metadata = response.json()

        # Save station metadata to temp directory
        temp_file_path = Path(self.extension_temp_dir_path) / "station_metadata.json"
        with open(temp_file_path, "w") as temp_file:
            json.dump(station_metadata, temp_file)

    def download_weather_data(self):
        """Download weather data for Mannheim stations."""
        self.logger.info("Downloading weather data for Mannheim stations...")
        # "https://stadtklimaanalyse-mannheim.de/wp-json/climate-data/v1/historic/{id}/{startdate}/{enddate}"
        base_url = (
            "https://stadtklimaanalyse-mannheim.de/wp-json/climate-data/v1/historic/"
        )

        years = range(self.start_datetime.year, self.end_datetime.year + 1)

        for weather_station in self.station_ids:
            station_data = {"data": {}}
            for year in years:
                self.logger.info(
                    f"Downloading year: {year} for station ID: {weather_station}"
                )
                year_start_date = datetime.datetime(year, 1, 1)
                year_end_date = datetime.datetime(year, 12, 31)
                if year == self.start_datetime.year:
                    year_start_date = self.start_datetime
                if year == self.end_datetime.year:
                    year_end_date = self.end_datetime
                for attempt in range(3):
                    try:
                        raw_data = requests.get(
                            f"{base_url}{weather_station}/{year_start_date.strftime('%Y-%m-%d')}/{year_end_date.strftime('%Y-%m-%d')}"
                        ).json()
                        break  # Break the loop if request is successful
                    except Exception as e:
                        self.logger.warning(
                            f"Attempt {attempt + 1} failed for station ID: {weather_station} for year {year}: {e}"
                        )
                        if attempt == 2:
                            self.logger.error(
                                f"Failed to download data for station ID: {weather_station} for year {year} after 3 attempts."
                            )
                            continue

                # needed data is under "data" key
                if "data" not in raw_data:
                    self.logger.warning(
                        f"No data found for station ID: {weather_station} for year {year}"
                    )
                    continue
                else:
                    # append data to station_data
                    station_data["data"].update(raw_data["data"])

            # Save raw data to temp directory
            temp_file_path = (
                Path(self.extension_temp_dir_path) / f"station_{weather_station}.json"
            )
            with open(temp_file_path, "w") as temp_file:
                json.dump(station_data, temp_file)

    def process_weather_data(self):
        """Process downloaded weather data from Mannheim stations."""
        self.logger.info("Processing weather data for Mannheim stations...")

        weather_data = []

        # Read downloaded JSON files and process data
        for weather_station in self.station_ids:
            with open(
                Path(self.extension_temp_dir_path) / "station_metadata.json",
                "r",
            ) as meta_file:
                station_metadata = json.load(meta_file)
                for station in station_metadata:
                    if str(station.get("station_id")) == str(weather_station):
                        lat = round(float(station.get("latitude")), 3)
                        lon = round(float(station.get("longitude")), 3)
                        self.logger.info(
                            f"Station ID: {weather_station}, Lat: {lat}, Lon: {lon}"
                        )

            with open(
                Path(self.extension_temp_dir_path) / f"station_{weather_station}.json",
                "r",
            ) as temp_file:
                station_data = json.load(temp_file)["data"]
                for key in station_data:
                    # 2023-01-04T00:00:00Z to datetime with timezone Europe/Berlin
                    # Parse UTC timestamp directly (Z suffix indicates UTC)
                    timestamp = datetime.datetime.strptime(
                        key, "%Y-%m-%dT%H:%M:%SZ"
                    ).replace(tzinfo=pytz.UTC)

                    temperature = (
                        round(float(station_data[key].get("t2m_med")), 2)
                        if station_data[key].get("t2m_med") is not None
                        else None
                    )
                    humidity = (
                        round(float(station_data[key].get("rf_med")), 2)
                        if station_data[key].get("rf_med") is not None
                        else None
                    )
                    precipitation = (
                        round(float(station_data[key].get("nied_med")), 2)
                        if station_data[key].get("nied_med") is not None
                        else None
                    )
                    wind_speed = (
                        round(float(station_data[key].get("wg_med")), 2)
                        if station_data[key].get("wg_med") is not None
                        else None
                    )
                    wind_direction = (
                        int(float(station_data[key].get("wr_med")))
                        if station_data[key].get("wr_med") is not None
                        else None
                    )
                    weather_data.append(
                        [
                            timestamp,
                            lat,
                            lon,
                            temperature,
                            humidity,
                            precipitation,
                            wind_speed,
                            wind_direction,
                        ]
                    )

        self.logger.info(f"Processed {len(weather_data)} weather data records")
        self.weather_data = weather_data

    def _get_unique_location_coordinates(self):
        """Extract unique coordinates and append to location_coordinates.parquet"""
        try:
            # Define the meta data file path first
            meta_dir = Path(self.meta_data_dir_path)
            meta_dir.mkdir(parents=True, exist_ok=True)
            coordinates_file = meta_dir / "location_coordinates.parquet"

            # Get unique coordinates from weather data
            unique_coords = set()
            for data_row in self.weather_data:
                lat = data_row[1]
                lng = data_row[2]
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
            for data_row in self.weather_data:
                lat = data_row[1]
                lng = data_row[2]

                # Get location ID from coordinates
                location_id = coord_to_id.get((lat, lng))
                if location_id is None:
                    continue

                # Initialize weather record with defaults
                weather_record = {
                    "location_id": location_id,
                    "timestamp": data_row[0],
                    "temperature": data_row[3],
                    "humidity": data_row[4],
                    "precipitation": data_row[5],
                    "wind_speed": data_row[6],
                    "wind_direction": data_row[7],
                }
                export_data.append(weather_record)

            location_ids = [r["location_id"] for r in export_data]
            timestamps = [r["timestamp"] for r in export_data]
            temperatures = [r["temperature"] for r in export_data]
            humidities = [r["humidity"] for r in export_data]
            precipitations = [r["precipitation"] for r in export_data]
            wind_speeds = [r["wind_speed"] for r in export_data]
            wind_directions = [r["wind_direction"] for r in export_data]

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

            # deduplicate weather_table
            weather_table = pa_ops.drop_duplicates(
                weather_table,
                ["location_id", "timestamp"],
                keep="first",
            )

            # sort weather_table by location_id and timestamp
            weather_table = weather_table.sort_by(
                [("location_id", "ascending"), ("timestamp", "ascending")]
            )

            # Write to parquet file
            output_file = (
                Path(self.extension_data_dir_path) / "mannheim_weather.parquet"
            )
            # If file exists, delete it first
            if output_file.exists():
                output_file.unlink()

            pq.write_table(weather_table, output_file, compression="BROTLI")

            # delete the tmp folder
            if Path(self.extension_temp_dir_path).exists():
                shutil.rmtree(Path(self.extension_temp_dir_path))

            self.logger.info(
                f"Exported {len(export_data)} weather records to {output_file}"
            )

        except Exception as e:
            self.logger.error(f"Error exporting weather data to parquet: {e}")
