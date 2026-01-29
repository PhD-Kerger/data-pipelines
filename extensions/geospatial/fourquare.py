from pathlib import Path
import requests
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import os
import warnings
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import tqdm

from utils import DataPipelineLogger

warnings.filterwarnings("ignore")


class Foursquare:
    def __init__(
        self,
        extension_data_dir_path,
        meta_data_dir_path,
        input_data_dir_path,
        logs_data_dir_path,
        cities,
        api_key,
    ):
        self.extension_data_dir_path = extension_data_dir_path + "/foursquare"
        self.meta_data_dir_path = meta_data_dir_path
        self.log_file = Path(logs_data_dir_path) / "logs.log"

        self.cities = cities if cities else []
        self.api_key = api_key
        self.pois = {city: [] for city in cities}

        # Setup logger as class attribute
        self.logger = DataPipelineLogger.get_logger(
            name=self.__class__.__name__, log_file_path=self.log_file
        )

        self.geodata_path = Path("./data/internal/geodata")
        self.plz_shapefile = f"{self.geodata_path}/plz-5stellig/plz-5stellig.shp"
        self.plz_ort = f"{self.geodata_path}/plz-ort/zuordnung_plz_ort.csv"

        self.plz = pd.read_csv(self.plz_ort)
        self.gdf = gpd.read_file(self.plz_shapefile)

        # Merge shapefile with PLZ info
        self.plz["plz"] = self.plz["plz"].astype(int)
        self.gdf["plz"] = self.gdf["plz"].astype(int)
        self.gdf = self.gdf.merge(self.plz, on="plz", how="left")

        # Reproject to meters
        self.gdf = self.gdf.to_crs(epsg=25832)

        os.makedirs(self.extension_data_dir_path, exist_ok=True)

    def run(self):
        """Main method to run Foursquare data processing"""
        if len(self.cities) == 0:
            self.logger.warning(
                "No cities provided for Foursquare extension. Terminating."
            )
            return

        for city in self.cities:
            points = self.generate_points(city)
            self.get_foursquare_pois(points, city)

        # Process and export data to parquet
        self._export_foursquare_data_to_parquet()

        self.logger.info("Foursquare processing completed")

    def get_foursquare_pois(self, points, city):
        """
        Fetches Foursquare POI data for given points and saves them to JSON files.
        """
        # limit for first 10 rows
        points = points.head(5)

        # Track FSQ IDs to prevent duplicates within the same city
        city_fsq_ids = set()

        for _, row in tqdm.tqdm(
            points.iterrows(), total=len(points), desc="Fetching Foursquare data"
        ):
            lat = row.geometry.y
            lng = row.geometry.x
            # check if file already exists
            if os.path.exists(f"{self.extension_data_dir_path}/{lat}_{lng}.json"):
                continue
            data = self.fetch_foursquare_data(lat, lng)

            if not data or "results" not in data:
                self.logger.warning(f"No Foursquare data found for {lat}, {lng}")
                continue

            for poi in data["results"]:
                fsq_id = poi.get("fsq_id")

                # Skip POIs without valid fsq_id or if already processed
                if not fsq_id or fsq_id in city_fsq_ids:
                    if fsq_id in city_fsq_ids:
                        self.logger.debug(
                            f"Skipping duplicate fsq_id for city {city}: {fsq_id}"
                        )
                    continue

                # Add to tracking set
                city_fsq_ids.add(fsq_id)

                self.pois[city].append(
                    {
                        # Foursquare specific fields
                        "fsq_id": fsq_id,
                        "name": poi.get("name"),
                        "latitude": poi.get("geocodes", {})
                        .get("main", {})
                        .get("latitude"),
                        "longitude": poi.get("geocodes", {})
                        .get("main", {})
                        .get("longitude"),
                        "categories": ", ".join(
                            [cat["name"] for cat in poi.get("categories", [])]
                        ),
                        # Business metrics
                        "popularity": poi.get("popularity", -1),
                        "rating": poi.get("rating", -1),
                        "price": poi.get("price", -1),
                        # Hours information
                        "hours_display": poi.get("hours", {}).get("display", ""),
                        # City information
                        "city": city,
                    }
                )

    def fetch_foursquare_data(self, lat, lng, radius=250, limit=50):
        """
        Fetches Foursquare data for a given latitude and longitude.
        """
        url = f"https://api.foursquare.com/v3/places/search"
        headers = {
            "Authorization": f"{self.api_key}",
            "accept": "application/json",
        }
        params = {
            "ll": f"{lat},{lng}",
            "radius": radius,
            "fields": "name,hours,hours_popular,rating,price,tastes,fsq_id,categories,popularity,geocodes",
            "limit": limit,
        }
        response = requests.get(url, headers=headers, params=params, verify=False)
        if response.status_code == 200:
            return response.json()
        else:
            self.logger.error(f"{response.status_code} - {response.text}")
            return None

    def generate_points(self, city):
        """
        Generates a grid of points within the boundaries of a given city,
        based on postal code areas (PLZ).
        """
        # Filter for the specified city
        gdf_city = self.gdf[self.gdf["ort"] == city]

        # Drop unused columns
        gdf_city = gdf_city.drop(
            columns=[
                "note",
                "einwohner",
                "qkm",
                "osm_id",
                "ags",
                "landkreis",
                "bundesland",
            ]
        )

        # Get bounds
        minx, miny, maxx, maxy = gdf_city.total_bounds

        # Grid spacing
        spacing = 250  # meters

        # Create meshgrid of coordinates
        x_coords = np.arange(minx, maxx, spacing)
        y_coords = np.arange(miny, maxy, spacing)
        xx, yy = np.meshgrid(x_coords, y_coords)

        # Flatten and create point geometries
        grid_points = [Point(x, y) for x, y in zip(xx.ravel(), yy.ravel())]

        # Create GeoDataFrame of points
        points_gdf = gpd.GeoDataFrame(geometry=grid_points, crs="EPSG:25832")

        # Spatial join: keep only points inside city polygons
        points_in_city = points_gdf[points_gdf.geometry.within(gdf_city.unary_union)]

        # Optional: convert to WGS84 for lat/lon use
        points_in_city = points_in_city.to_crs(epsg=4326)

        return points_in_city

    def _get_unique_location_coordinates(self):
        """Extract unique coordinates and append to location_coordinates.parquet"""
        try:
            # Define the meta data file path
            meta_dir = Path(self.meta_data_dir_path)
            meta_dir.mkdir(parents=True, exist_ok=True)
            coordinates_file = meta_dir / "location_coordinates.parquet"

            # Get unique coordinates from Foursquare data
            unique_coords = set()
            for city in self.cities:
                for poi in self.pois.get(city, []):
                    if poi.get("latitude") and poi.get("longitude"):
                        lat = round(float(poi["latitude"]), 3)
                        lng = round(float(poi["longitude"]), 3)
                        unique_coords.add((lat, lng))

            if not unique_coords:
                self.logger.warning("No coordinates found in Foursquare data")
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
                    pq.write_table(combined_data, coordinates_file, compression="BROTLI")
                else:
                    pq.write_table(new_data, coordinates_file, compression="BROTLI")

                self.logger.info(f"Saved coordinates to {coordinates_file}")

        except Exception as e:
            self.logger.error(f"Error extracting coordinates: {e}")

    def _map_coordinates_to_location_ids(self):
        """Map coordinates in Foursquare data to location IDs from the parquet file"""
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

    def _export_foursquare_data_to_parquet(self):
        """Export Foursquare data to parquet file with proper schema"""
        try:
            # First ensure location coordinates are up to date
            self._get_unique_location_coordinates()

            # Get coordinate to location ID mapping
            coord_to_id = self._map_coordinates_to_location_ids()
            if not coord_to_id:
                self.logger.error("No coordinate mappings available")
                return

            # Check for existing Foursquare data to avoid duplicates
            output_file = Path(self.extension_data_dir_path) / "foursquare.parquet"
            existing_fsq_ids = set()

            if output_file.exists():
                try:
                    existing_data = pq.read_table(output_file)
                    # Check if the file has data and the expected columns
                    if (
                        len(existing_data) > 0
                        and "fsq_id" in existing_data.column_names
                    ):
                        existing_fsq_ids = set(
                            existing_data.column("fsq_id").to_pylist()
                        )
                        # Remove None values from the set
                        existing_fsq_ids.discard(None)
                        self.logger.info(
                            f"Found {len(existing_fsq_ids)} existing Foursquare IDs"
                        )
                except Exception as e:
                    self.logger.warning(f"Could not read existing Foursquare data: {e}")
                    existing_fsq_ids = set()

            # Prepare data for export
            export_data = []
            # Track FSQ IDs in current batch to prevent duplicates within the same run
            current_batch_fsq_ids = set()

            for city in self.cities:
                pois = self.pois.get(city, [])

                for poi in pois:
                    if not poi.get("latitude") or not poi.get("longitude"):
                        continue

                    # Skip if this Foursquare ID already exists
                    fsq_id = poi.get("fsq_id")
                    if not fsq_id:  # Skip POIs without valid fsq_id
                        continue

                    if fsq_id in existing_fsq_ids:
                        continue

                    # Skip if this FSQ ID already exists in current batch
                    if fsq_id in current_batch_fsq_ids:
                        self.logger.debug(
                            f"Skipping duplicate fsq_id in current batch: {fsq_id}"
                        )
                        continue

                    lat = round(float(poi["latitude"]), 3)
                    lng = round(float(poi["longitude"]), 3)

                    # Get location ID from coordinates
                    location_id = coord_to_id.get((lat, lng))
                    if location_id is None:
                        continue

                    # Create Foursquare record
                    foursquare_record = {
                        "fsq_id": fsq_id,
                        "name": poi.get("name", "")[:200],  # Truncate to 200 chars
                        "location_id": location_id,
                        "categories": poi.get("categories", "")[
                            :200
                        ],  # Truncate to 200 chars
                        "popularity": (
                            float(poi.get("popularity", -1))
                            if poi.get("popularity", -1) != -1
                            else None
                        ),
                        "rating": (
                            float(poi.get("rating", -1))
                            if poi.get("rating", -1) != -1
                            else None
                        ),
                        "price": (
                            float(poi.get("price", -1))
                            if poi.get("price", -1) != -1
                            else None
                        ),
                        "hours_display": (
                            poi.get("hours_display", "")[:200]
                            if poi.get("hours_display")
                            else ""
                        ),  # Truncate to 200 chars
                    }

                    export_data.append(foursquare_record)
                    # Add to current batch tracking
                    current_batch_fsq_ids.add(fsq_id)

            if not export_data:
                self.logger.warning("No new Foursquare data to export")
                return

            # Convert to lists for PyArrow
            fsq_ids = [r["fsq_id"] for r in export_data]
            names = [r["name"] for r in export_data]
            location_ids = [r["location_id"] for r in export_data]
            categories = [r["categories"] for r in export_data]
            popularities = [r["popularity"] for r in export_data]
            ratings = [r["rating"] for r in export_data]
            prices = [r["price"] for r in export_data]
            hours_displays = [r["hours_display"] for r in export_data]

            # Create PyArrow table
            new_foursquare_table = pa.table(
                {
                    "fsq_id": pa.array(fsq_ids, type=pa.string()),
                    "name": pa.array(names, type=pa.string()),
                    "location_id": pa.array(location_ids, type=pa.int32()),
                    "categories": pa.array(categories, type=pa.string()),
                    "popularity": pa.array(popularities, type=pa.float64()),
                    "rating": pa.array(ratings, type=pa.float64()),
                    "price": pa.array(prices, type=pa.float64()),
                    "hours_display": pa.array(hours_displays, type=pa.string()),
                }
            )

            # Write to parquet file (append if exists, create if not)
            if output_file.exists():
                # Read existing data and append new data
                existing_data = pq.read_table(output_file)
                combined_data = pa.concat_tables([existing_data, new_foursquare_table])
                pq.write_table(combined_data, output_file, compression="BROTLI")
                self.logger.info(
                    f"Appended {len(export_data)} new Foursquare records to {output_file}"
                )
            else:
                pq.write_table(new_foursquare_table, output_file, compression="BROTLI")
                self.logger.info(
                    f"Created new file with {len(export_data)} Foursquare records at {output_file}"
                )

        except Exception as e:
            self.logger.error(f"Error exporting Foursquare data to parquet: {e}")
