import os
import shutil
from pathlib import Path

import openmeteo_requests
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow_ops as pa_ops
import requests_cache
from retry_requests import retry

from utils.data_pipeline_logger import DataPipelineLogger


class OpenMeteo:
    """Extension for Open-Meteo weather data."""

    def __init__(
        self,
        extension_data_dir_path,
        meta_data_dir_path,
        input_data_dir_path,
        logs_data_dir_path,
        start_date,
        end_date,
        locations,
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
        # locations is a list of [lat, lon] pairs
        self.locations = locations if locations else []

        # Setup the Open-Meteo API client with cache and retry on error
        self.cache_session = requests_cache.CachedSession(
            str(Path(self.extension_temp_dir_path) / ".cache"), expire_after=-1
        )
        self.retry_session = retry(self.cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=self.retry_session)

        self.weather_data = []

    @property
    def start_date_formatted(self):
        """Convert YYYYMMDD to YYYY-MM-DD for Open-Meteo API"""
        return f"{self.start_date[:4]}-{self.start_date[4:6]}-{self.start_date[6:8]}"

    @property
    def end_date_formatted(self):
        """Convert YYYYMMDD to YYYY-MM-DD for Open-Meteo API"""
        return f"{self.end_date[:4]}-{self.end_date[4:6]}-{self.end_date[6:8]}"

    def run(self):
        """Main method to run the Open-Meteo weather data processing."""
        if not self.locations:
            self.logger.warning(
                "No locations provided for Open-Meteo extension. Terminating."
            )
            return

        self.logger.info(
            f"Starting Open-Meteo weather data processing for {len(self.locations)} locations"
        )

        # Step 1: Download and process weather data
        self.fetch_weather_data()

        # Step 2: Extract unique coordinates and update metadata
        self._get_unique_location_coordinates()

        # Step 3: Export weather data to parquet format
        self.export_weather_data_to_parquet()

        self.logger.info("Open-Meteo weather processing completed")

    def fetch_weather_data(self):
        """Fetch weather data from Open-Meteo API for all locations in batches."""
        url = "https://archive-api.open-meteo.com/v1/archive"

        self.weather_data = []
        batch_size = 100

        for i in range(0, len(self.locations), batch_size):
            batch = self.locations[i : i + batch_size]
            lats = [loc[0] for loc in batch]
            lons = [loc[1] for loc in batch]

            self.logger.info(
                f"Fetching data for batch of {len(batch)} locations (starting index {i})"
            )

            params = {
                "latitude": lats,
                "longitude": lons,
                "start_date": self.start_date_formatted,
                "end_date": self.end_date_formatted,
                "hourly": [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "precipitation",
                    "wind_speed_10m",
                    "wind_direction_10m",
                ],
                "timezone": "GMT",
                "timeformat": "unixtime",
            }

            try:
                responses = self.openmeteo.weather_api(url, params=params)

                for idx, response in enumerate(responses):
                    lat = lats[idx]
                    lon = lons[idx]

                    hourly = response.Hourly()

                    # Check if we have the expected number of variables
                    if hourly.VariablesLength() < 5:
                        self.logger.warning(
                            f"Unexpected number of variables for {lat}, {lon}"
                        )
                        continue

                    # The order of variables needs to be the same as requested.
                    # We use PyArrow's timestamp construction since we've removed pandas
                    start_time = hourly.Time()
                    interval = hourly.Interval()
                    num_values = len(hourly.Variables(0).ValuesAsNumpy())

                    # Generate timestamps using range and simple arithmetic
                    # open-meteo returns unix timestamps (seconds)
                    timestamps = [
                        start_time + (j * interval) for j in range(num_values)
                    ]

                    temps = hourly.Variables(0).ValuesAsNumpy()
                    humidities = hourly.Variables(1).ValuesAsNumpy()
                    precips = hourly.Variables(2).ValuesAsNumpy()
                    wind_speeds = hourly.Variables(3).ValuesAsNumpy()
                    wind_dirs = hourly.Variables(4).ValuesAsNumpy()

                    for j in range(num_values):
                        self.weather_data.append(
                            {
                                "timestamp": timestamps[
                                    j
                                ],  # Int, will be converted to pa.timestamp later
                                "temperature": float(temps[j]),
                                "humidity": float(humidities[j]),
                                "precipitation": float(precips[j]),
                                "wind_speed": float(wind_speeds[j]),
                                "wind_direction": int(wind_dirs[j]),
                                "lat": round(float(lat), 3),
                                "lng": round(float(lon), 3),
                            }
                        )

            except Exception as e:
                self.logger.error(f"Error fetching data for batch starting at {i}: {e}")

    def _get_unique_location_coordinates(self):
        """Extract unique coordinates and append to location_coordinates.parquet"""
        try:
            meta_dir = Path(self.meta_data_dir_path)
            meta_dir.mkdir(parents=True, exist_ok=True)
            coordinates_file = meta_dir / "location_coordinates.parquet"

            unique_coords = set()
            for record in self.weather_data:
                unique_coords.add((record["lat"], record["lng"]))

            if not unique_coords:
                self.logger.warning("No coordinates found in weather data")
                return

            new_coords = list(unique_coords)

            if coordinates_file.exists():
                existing_data = pq.read_table(coordinates_file)
                existing_coords = set(
                    zip(
                        existing_data.column("lat").to_pylist(),
                        existing_data.column("lng").to_pylist(),
                    )
                )
                max_location_id = (
                    max(existing_data.column("location_id").to_pylist())
                    if existing_data.num_rows > 0
                    else 0
                )

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

            # Create new location IDs
            location_ids = []
            lats = []
            lngs = []
            for i, (lat, lng) in enumerate(new_coords):
                location_ids.append(max_location_id + i + 1)
                lats.append(lat)
                lngs.append(lng)

            new_table = pa.table(
                {
                    "location_id": pa.array(location_ids, type=pa.int32()),
                    "lat": pa.array(lats, type=pa.float64()),
                    "lng": pa.array(lngs, type=pa.float64()),
                }
            )

            if coordinates_file.exists():
                existing_table = pq.read_table(coordinates_file)
                # Combine tables
                combined_table = pa.concat_tables([existing_table, new_table])
                pq.write_table(combined_table, coordinates_file)
            else:
                pq.write_table(new_table, coordinates_file)

        except Exception as e:
            self.logger.error(f"Error updating location coordinates: {e}")

    def _map_coordinates_to_location_ids(self):
        """Map coordinates in weather data to location IDs from the parquet file"""
        try:
            meta_dir = Path(self.meta_data_dir_path)
            coordinates_file = meta_dir / "location_coordinates.parquet"

            if not coordinates_file.exists():
                self.logger.error("location_coordinates.parquet not found.")
                return {}

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
        """Export weather data to parquet file with proper schema, joining, deduplication and sorting."""
        try:
            coord_to_id = self._map_coordinates_to_location_ids()
            if not coord_to_id:
                self.logger.error("No coordinate mappings available")
                return

            final_records = []
            for record in self.weather_data:
                location_id = coord_to_id.get((record["lat"], record["lng"]))
                if location_id is None:
                    continue

                final_records.append(
                    {
                        "location_id": location_id,
                        "timestamp": record["timestamp"],
                        "temperature": record["temperature"],
                        "humidity": record["humidity"],
                        "precipitation": record["precipitation"],
                        "wind_speed": record["wind_speed"],
                        "wind_direction": record["wind_direction"],
                    }
                )

            if not final_records:
                self.logger.warning("No weather data matches known location IDs")

            new_weather_table = pa.table(
                {
                    "location_id": pa.array(
                        [r["location_id"] for r in final_records], type=pa.int32()
                    ),
                    "timestamp": pa.array(
                        [r["timestamp"] * 1000 for r in final_records],
                        type=pa.timestamp("ms", tz="UTC"),
                    ),
                    "temperature": pa.array(
                        [r["temperature"] for r in final_records], type=pa.float64()
                    ),
                    "humidity": pa.array(
                        [r["humidity"] for r in final_records], type=pa.float64()
                    ),
                    "precipitation": pa.array(
                        [r["precipitation"] for r in final_records], type=pa.float64()
                    ),
                    "wind_speed": pa.array(
                        [r["wind_speed"] for r in final_records], type=pa.float64()
                    ),
                    "wind_direction": pa.array(
                        [r["wind_direction"] for r in final_records], type=pa.int32()
                    ),
                }
            )

            output_file = Path(self.extension_data_dir_path) / "openmeteo.parquet"

            if output_file.exists():
                existing_table = pq.read_table(output_file)
                weather_table = pa.concat_tables([existing_table, new_weather_table])
                self.logger.info(
                    f"Joined {new_weather_table.num_rows} new records with {existing_table.num_rows} existing records"
                )
            else:
                weather_table = new_weather_table
                self.logger.info(
                    f"Creating new weather data file with {weather_table.num_rows} records"
                )

            # Deduplicate and sort
            if weather_table.num_rows > 0:
                weather_table = pa_ops.drop_duplicates(
                    weather_table,
                    ["location_id", "timestamp"],
                    keep="first",
                )

                # Sort by location_id and timestamp
                weather_table = weather_table.sort_by(
                    [("location_id", "ascending"), ("timestamp", "ascending")]
                )

            pq.write_table(weather_table, output_file, compression="BROTLI")

            # Clean up tmp folder
            if Path(self.extension_temp_dir_path).exists():
                shutil.rmtree(Path(self.extension_temp_dir_path))

            self.logger.info(
                f"Final dataset has {weather_table.num_rows} records. Saved to {output_file}"
            )

        except Exception as e:
            self.logger.error(f"Error exporting weather data: {e}")
