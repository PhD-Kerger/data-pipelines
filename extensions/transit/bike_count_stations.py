from pathlib import Path
import requests
import pyarrow as pa
import pyarrow.parquet as pq
import datetime
import json
import os
import tempfile

import gzip
from utils import DataPipelineLogger


class BikeCountStationsGermany:
    def __init__(
        self,
        extension_data_dir_path,
        meta_data_dir_path,
        logs_data_dir_path,
        input_data_dir_path,
        start_date,
        end_date,
        domain_names,
    ):
        self.extension_data_dir_path = (
            Path(extension_data_dir_path) / "bike_count_stations" / "germany"
        )
        self.meta_data_dir_path = Path(meta_data_dir_path)
        self.log_file = Path(logs_data_dir_path) / "logs.log"
        self.start_date = start_date
        self.end_date = end_date
        self.domain_names = domain_names
        self.base_url = (
            "https://mobidata-bw.de/fahrradzaehldaten/v2/fahrradzaehler_stundenwerten"
        )

        # Setup logger
        self.logger = DataPipelineLogger.get_logger(
            name=self.__class__.__name__, log_file_path=self.log_file
        )

        self.extension_data_dir_path.mkdir(parents=True, exist_ok=True)

        self.data = []

    def run(self):
        """Main method to run the BikeCountStationsGermany extension"""
        if len(self.domain_names) == 0:
            self.logger.warning("No domain names provided, skipping extension")
            return

        # Step 1: Get bike count stations data
        self.get_bike_count_stations()
        # Step 2: Export data to parquet
        self._export_bike_count_data_to_parquet()

    def get_bike_count_stations(self):
        """Get bike count stations in Germany from mobidata-bw.de for the given date range"""

        start_year = int(self.start_date[:4])
        start_month = int(self.start_date[4:6])
        end_year = int(self.end_date[:4])
        end_month = int(self.end_date[4:6])

        date_range = []
        current_year = start_year
        current_month = start_month
        while (current_year < end_year) or (
            current_year == end_year and current_month <= end_month
        ):
            date_range.append((current_year, current_month))
            if current_month == 12:
                current_month = 1
                current_year += 1
            else:
                current_month += 1

        self.logger.info(f"Fetching bike count stations for {len(date_range)} months")

        for year, month in date_range:
            records = []
            url = f"{self.base_url}_{year}{month:02d}.json.gz"
            response = requests.get(url, verify=False)
            if response.status_code == 200:
                data = response.content
                with tempfile.NamedTemporaryFile(
                    suffix=".gz", delete=False
                ) as tmp_file:
                    tmp_file.write(data)
                    tmp_file_path = tmp_file.name
                # Read the gzipped json file
                with gzip.open(tmp_file_path, "rt", encoding="utf-8") as f:
                    data = json.load(f)
                for entry in data:
                    domain_name = entry["domain_name"]
                    if domain_name in self.domain_names:
                        counter_site = entry["counter_site"]
                        counter_site_id = entry["counter_site_id"]
                        longitude = round(entry["longitude"], 3)
                        latitude = round(entry["latitude"], 3)
                        channels = entry["channels"]

                        # Group channels by timestamp
                        timestamp_data = {}
                        for channel in channels:
                            timestamp = channel["iso_timestamp"]
                            direction = channel["direction"]

                            if direction in ["IN", "OUT"]:
                                if timestamp not in timestamp_data:
                                    timestamp_data[timestamp] = {
                                        "counter_site_id": counter_site_id,
                                        "counter_site": counter_site,
                                        "longitude": longitude,
                                        "latitude": latitude,
                                        "timestamp": timestamp,
                                        "in": None,
                                        "out": None,
                                    }

                                if direction == "IN":
                                    timestamp_data[timestamp]["in"] = channel["counts"]
                                elif direction == "OUT":
                                    timestamp_data[timestamp]["out"] = channel["counts"]

                        # Process each timestamp entry
                        for timestamp, data in timestamp_data.items():
                            record = {
                                "counter_id": data["counter_site_id"],
                                "counter_name": data["counter_site"],
                                "longitude": data["longitude"],
                                "latitude": data["latitude"],
                                "timestamp": datetime.datetime.fromisoformat(
                                    data["timestamp"]
                                ),
                                "in_count": data["in"],
                                "out_count": data["out"],
                            }
                            records.append(record)
                os.remove(tmp_file_path)
                self.logger.info(
                    f"Fetched {len(records)} records for {year}-{month:02d}"
                )
                self.data.extend(records)
            else:
                self.logger.warning(
                    f"Failed to fetch data from {url}, status code: {response.status_code}"
                )

    def _get_unique_location_coordinates(self):
        """Extract unique coordinates and append to location_coordinates.parquet"""
        try:
            # Define the meta data file path
            meta_dir = Path(self.meta_data_dir_path)
            meta_dir.mkdir(parents=True, exist_ok=True)
            coordinates_file = meta_dir / "location_coordinates.parquet"

            # Get unique coordinates from Bike Counter data
            unique_coords = set()
            for entry in self.data:
                lat = round(float(entry["latitude"]), 3)
                lng = round(float(entry["longitude"]), 3)
                unique_coords.add((lat, lng))
            if not unique_coords:
                self.logger.warning("No coordinates found in Bike Counter data")
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
        """Map coordinates in OSM data to location IDs from the parquet file"""
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

    def _export_bike_count_data_to_parquet(self):
        """Export Bike count data to parquet file with proper schema"""
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

            for entry in self.data:
                lat = round(float(entry["latitude"]), 3)
                lng = round(float(entry["longitude"]), 3)

                # Get location ID from coordinates
                location_id = coord_to_id.get((lat, lng))
                if location_id is None:
                    continue

                # Create bike count record
                bike_count_record = {
                    "counter_id": entry["counter_id"],
                    "counter_name": entry["counter_name"],
                    "location_id": location_id,
                    "timestamp": entry["timestamp"],
                    "in_count": entry["in_count"],
                    "out_count": entry["out_count"],
                }

                export_data.append(bike_count_record)

            if not export_data:
                self.logger.warning("No OSM data to export")
                return

            # Convert to lists for PyArrow
            counter_ids = [d["counter_id"] for d in export_data]
            counter_names = [d["counter_name"] for d in export_data]
            location_ids = [d["location_id"] for d in export_data]
            timestamps = [d["timestamp"] for d in export_data]
            ins = [d["in_count"] for d in export_data]
            outs = [d["out_count"] for d in export_data]

            schema = pa.schema(
                [
                    pa.field("counter_id", pa.int32()),
                    pa.field("timestamp", pa.timestamp("s", tz="Europe/Berlin")),
                    pa.field("location_id", pa.int32()),
                    pa.field("counter_name", pa.string()),
                    pa.field("in_count", pa.int32()),
                    pa.field("out_count", pa.int32()),
                ]
            )

            osm_table = pa.table(
                {
                    "counter_id": pa.array(counter_ids, type=pa.int32()),
                    "timestamp": pa.array(
                        timestamps, type=pa.timestamp("s", tz="Europe/Berlin")
                    ),
                    "location_id": pa.array(location_ids, type=pa.int32()),
                    "counter_name": pa.array(counter_names, type=pa.string()),
                    "in_count": pa.array(ins, type=pa.int32()),
                    "out_count": pa.array(outs, type=pa.int32()),
                },
                schema=schema,
            )

            # Write to parquet file
            output_file = Path(self.extension_data_dir_path) / "bike_counts.parquet"

            pq.write_table(osm_table, output_file, compression="BROTLI")

            self.logger.info(
                f"Exported {len(export_data)} bike count records to {output_file}"
            )

        except Exception as e:
            self.logger.error(f"Error exporting bike count data to parquet: {e}")
