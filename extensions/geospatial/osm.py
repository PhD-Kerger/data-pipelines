from OSMPythonTools.nominatim import Nominatim
import datetime
import overpy
import time
import os
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

from utils import DataPipelineLogger


class OSM:
    # Configuration for each OSM entity type
    OSM_CONFIG = {
        "restaurant": {
            "osm_key": "amenity",
            "osm_value": "restaurant",
            "tags": ["name", "cuisine", "opening_hours"],
        },
        "bar": {
            "osm_key": "amenity",
            "osm_value": "bar",
            "tags": ["name", "opening_hours"],
        },
        "bakery": {
            "osm_key": "amenity",
            "osm_value": "bakery",
            "tags": ["name", "opening_hours"],
        },
        "cafe": {
            "osm_key": "amenity",
            "osm_value": "cafe",
            "tags": ["name", "opening_hours"],
        },
        "fast_food": {
            "osm_key": "amenity",
            "osm_value": "fast_food",
            "tags": ["name", "cuisine", "opening_hours"],
        },
        "university": {
            "osm_key": "amenity",
            "osm_value": "university",
            "tags": ["name"],
        },
        "college": {"osm_key": "amenity", "osm_value": "college", "tags": ["name"]},
        "school": {"osm_key": "amenity", "osm_value": "school", "tags": ["name"]},
        "kindergarten": {
            "osm_key": "amenity",
            "osm_value": "kindergarten",
            "tags": ["name"],
        },
        "hospital": {
            "osm_key": "amenity",
            "osm_value": "hospital",
            "tags": ["name", "opening_hours"],
        },
        "library": {
            "osm_key": "amenity",
            "osm_value": "library",
            "tags": ["name", "opening_hours"],
        },
        "place_of_worship": {
            "osm_key": "amenity",
            "osm_value": "place_of_worship",
            "tags": ["name"],
        },
        "marketplace": {
            "osm_key": "amenity",
            "osm_value": "marketplace",
            "tags": ["name", "opening_hours"],
        },
        "bank": {
            "osm_key": "amenity",
            "osm_value": "bank",
            "tags": ["name", "opening_hours"],
        },
        "doctors": {
            "osm_key": "amenity",
            "osm_value": "doctors",
            "tags": ["name", "opening_hours"],
        },
        "pharmacy": {
            "osm_key": "amenity",
            "osm_value": "pharmacy",
            "tags": ["name", "opening_hours"],
        },
        "dentist": {
            "osm_key": "amenity",
            "osm_value": "dentist",
            "tags": ["name", "opening_hours"],
        },
        "chemist": {
            "osm_key": "amenity",
            "osm_value": "chemist",
            "tags": ["name", "opening_hours"],
        },
        "cinema": {"osm_key": "amenity", "osm_value": "cinema", "tags": ["name"]},
        "clothes": {
            "osm_key": "shop",
            "osm_value": "clothes",
            "tags": ["name", "opening_hours"],
        },
        "department_store": {
            "osm_key": "shop",
            "osm_value": "department_store",
            "tags": ["name", "opening_hours"],
        },
        "parcel_locker": {
            "osm_key": "amenity",
            "osm_value": "parcel_locker",
            "tags": ["name", "opening_hours"],
        },
        "supermarket": {
            "osm_key": "shop",
            "osm_value": "supermarket",
            "tags": ["name", "opening_hours"],
        },
        "theatre": {
            "osm_key": "amenity",
            "osm_value": "theatre",
            "tags": ["name", "opening_hours"],
        },
        "townhall": {"osm_key": "amenity", "osm_value": "townhall", "tags": ["name"]},
        "public_transport_platform": {
            "osm_key": "public_transport",
            "osm_value": "platform",
            "tags": ["name"],
        },
        "tram_stop": {"osm_key": "railway", "osm_value": "tram_stop", "tags": ["name"]},
        "station": {
            "osm_key": "public_transport",
            "osm_value": "station",
            "tags": ["name"],
        },
        "bus_stop": {"osm_key": "highway", "osm_value": "bus_stop", "tags": ["name"]},
        "attraction": {
            "osm_key": "tourism",
            "osm_value": "attraction",
            "tags": ["name"],
        },
        "museum": {"osm_key": "tourism", "osm_value": "museum", "tags": ["name"]},
        "park": {"osm_key": "leisure", "osm_value": "park", "tags": ["name"]},
    }

    def __init__(
        self,
        extension_data_dir_path,
        meta_data_dir_path,
        logs_data_dir_path,
        input_data_dir_path,
        years,
        cities,
        tags,
    ):
        self.extension_data_dir_path = extension_data_dir_path + "/osm"
        self.meta_data_dir_path = meta_data_dir_path
        self.log_file = Path(logs_data_dir_path) / "logs.log"
        self.years = years
        self.cities = cities if cities else []
        self.tags = tags if tags else []

        self.area_ids = []  # List to store area IDs from OSM
        self.osm_entities = {tag: [] for tag in self.tags}

        # Setup logger
        self.logger = DataPipelineLogger.get_logger(
            name=self.__class__.__name__, log_file_path=self.log_file
        )

        os.makedirs(self.extension_data_dir_path, exist_ok=True)

    def run(self):
        """Main method to run OSM data processing"""
        if len(self.cities) == 0:
            self.logger.warning("No cities provided for OSM extension. Terminating.")
            return

        if len(self.tags) == 0:
            self.logger.warning("No tags provided for OSM extension. Terminating.")
            return

        # Step 1: Get area IDs for cities
        self._get_area_ids()

        # Step 2: Get OSM entities for all tags
        self._get_osm_entities()

        # Step 3: Export OSM data to parquet format
        self._export_osm_data_to_parquet()

        self.logger.info("OSM processing completed")

    def _get_area_ids(self):
        """Get the area ids for each city"""
        nominatim = Nominatim()

        for city in self.cities:
            try:
                result = nominatim.query(city)
                if result:
                    self.area_ids.append({"city": city, "area_id": result.areaId()})
                else:
                    self.logger.warning(f"Could not find area ID for {city}")
            except Exception as e:
                self.logger.error(f"Error getting area ID for {city}: {e}")

    def _get_osm_entities(self):
        """Get OSM entities for all configured tags"""
        api = overpy.Overpass(url="https://overpass.kumi.systems/api/interpreter")

        for tag in self.tags:
            if tag not in self.OSM_CONFIG:
                self.logger.warning(f"Tag '{tag}' not found in OSM_CONFIG. Skipping.")
                continue

            config = self.OSM_CONFIG[tag]
            self.logger.info(f"Processing OSM data for {tag}")

            for area in self.area_ids:
                city = area["city"]
                area_id = area["area_id"]

                for year in self.years:
                    self.logger.info(f"Fetching {tag} data for {city} in {year}")
                    time.sleep(1.5)  # Sleep to respect API rate limits
                    try:
                        # Build OSM query
                        query = f"""
                        [out:json];
                        area({area_id});
                        node(area)["{config['osm_key']}"="{config['osm_value']}"];
                        out;
                        """
                        result = api.query(query)

                        for node in result.nodes:
                            # Create base entity data
                            entity_data = {
                                "city": city,
                                "timestamp": int(
                                    datetime.datetime.strptime(
                                        f"{year}-01-01T12:00:00Z", "%Y-%m-%dT%H:%M:%SZ"
                                    ).timestamp()
                                ),
                                "osm_id": node.id,
                                "lat": round(node.lat, 4),
                                "lon": round(node.lon, 4),
                                "entity_name": tag,
                            }

                            # Add configured tags
                            for tag_name in config["tags"]:
                                entity_data[tag_name] = node.tags.get(tag_name, "")

                            # Ensure all possible fields are present
                            for field in ["name", "cuisine", "opening_hours"]:
                                if field not in entity_data:
                                    entity_data[field] = ""

                            self.osm_entities[tag].append(entity_data)

                        self.logger.info(
                            f"Fetched {len(result.nodes)} {tag} entities for {city} in {year}"
                        )

                    except Exception as e:
                        self.logger.error(
                            f"Error processing {tag} for {city} in {year}: {e}"
                        )
                        # add a info line in a additional log file to track which city and year had issues
                        with open(
                            Path(self.extension_data_dir_path) / "osm_errors.log", "a"
                        ) as error_log:
                            error_log.write(f"{tag} - {city}\n")
                        continue

        self.logger.info("OSM data collection completed")

    def _get_unique_location_coordinates(self):
        """Extract unique coordinates and append to location_coordinates.parquet"""
        try:
            # Define the meta data file path
            meta_dir = Path(self.meta_data_dir_path)
            meta_dir.mkdir(parents=True, exist_ok=True)
            coordinates_file = meta_dir / "location_coordinates.parquet"

            # Get unique coordinates from OSM data
            unique_coords = set()
            for tag in self.tags:
                for entity in self.osm_entities.get(tag, []):
                    lat = round(float(entity["lat"]), 3)
                    lng = round(float(entity["lon"]), 3)
                    unique_coords.add((lat, lng))

            if not unique_coords:
                self.logger.warning("No coordinates found in OSM data")
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

    def _export_osm_data_to_parquet(self):
        """Export OSM data to parquet file with proper schema"""
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
            osm_id_counter = 1  # Auto-increment ID for OSM entities

            # check if there is already data available
            existing_file = Path(self.extension_data_dir_path) / "osm.parquet"
            if existing_file.exists():
                self.logger.info("OSM parquet file already exists. Merging new data.")
                # read existing data
                existing_table = pq.read_table(
                    existing_file,
                    schema=pa.schema(
                        [
                            pa.field("id", pa.int32()),
                            pa.field("timestamp", pa.timestamp("s", tz="UTC")),
                            pa.field("location_id", pa.int32()),
                            pa.field("entity_name", pa.string()),
                            pa.field("name", pa.string()),
                            pa.field("cuisine", pa.string()),
                            pa.field("opening_hours", pa.string()),
                        ]
                    ),
                )

                # get max id
                existing_ids = existing_table.column("id").to_pylist()
                if existing_ids:
                    osm_id_counter = max(existing_ids) + 1

            for tag in self.tags:
                entities = self.osm_entities.get(tag, [])

                for entity in entities:
                    lat = round(float(entity["lat"]), 3)
                    lng = round(float(entity["lon"]), 3)

                    # Get location ID from coordinates
                    location_id = coord_to_id.get((lat, lng))
                    if location_id is None:
                        continue

                    # Create OSM record
                    osm_record = {
                        "id": osm_id_counter,
                        "timestamp": entity["timestamp"],
                        "location_id": location_id,
                        "entity_name": entity["entity_name"],
                        "name": entity.get("name", ""),
                        "cuisine": entity.get("cuisine", ""),
                        "opening_hours": entity.get("opening_hours", ""),
                    }

                    export_data.append(osm_record)
                    osm_id_counter += 1

            if not export_data:
                self.logger.warning("No OSM data to export")
                return

            # Convert to lists for PyArrow
            ids = [r["id"] for r in export_data]
            timestamps = [r["timestamp"] for r in export_data]
            location_ids = [r["location_id"] for r in export_data]
            entity_names = [r["entity_name"] for r in export_data]
            names = [r["name"] for r in export_data]
            cuisines = [r["cuisine"] for r in export_data]
            opening_hours = [r["opening_hours"] for r in export_data]

            schema = pa.schema(
                [
                    pa.field("id", pa.int32()),
                    pa.field("timestamp", pa.timestamp("s", tz="UTC")),
                    pa.field("location_id", pa.int32()),
                    pa.field("entity_name", pa.string()),
                    pa.field("name", pa.string()),
                    pa.field("cuisine", pa.string()),
                    pa.field("opening_hours", pa.string()),
                ]
            )

            osm_table = pa.table(
                {
                    "id": pa.array(ids, type=pa.int32()),
                    "timestamp": pa.array(timestamps, type=pa.timestamp("s", tz="UTC")),
                    "location_id": pa.array(location_ids, type=pa.int32()),
                    "entity_name": pa.array(entity_names, type=pa.string()),
                    "name": pa.array(names, type=pa.string()),
                    "cuisine": pa.array(cuisines, type=pa.string()),
                    "opening_hours": pa.array(opening_hours, type=pa.string()),
                },
                schema=schema,
            )

            # Write to parquet file
            output_file = Path(self.extension_data_dir_path) / "osm.parquet"

            if existing_file.exists():
                # Merge with existing data
                combined_table = pa.concat_tables([existing_table, osm_table])
                pq.write_table(combined_table, output_file, compression="BROTLI")
            else:
                pq.write_table(osm_table, output_file, compression="BROTLI")

            self.logger.info(
                f"Exported {len(export_data)} OSM records to {output_file}"
            )

        except Exception as e:
            self.logger.error(f"Error exporting OSM data to parquet: {e}")
