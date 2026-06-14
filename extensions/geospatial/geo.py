import os
from pathlib import Path
from shapely.geometry import Point
import duckdb
import requests
import pyarrow as pa
import pyarrow.parquet as pq
import geopandas as gpd
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import json

from utils import DataPipelineLogger


class Geo:
    def __init__(
        self,
        extension_data_dir_path,
        meta_data_dir_path,
        input_data_dir_path,
        logs_data_dir_path,
        enable_elevation,
        elevation_dataset,
        elevation_endpoint,
    ):
        self.extension_data_dir_path = extension_data_dir_path + "/geo"
        self.meta_data_dir_path = meta_data_dir_path
        self.log_file = Path(logs_data_dir_path) / "logs.log"

        # Load shapefiles with proper paths
        geodata_path = Path("./data/internal/geodata")
        self.SPATIAL_SHP = gpd.read_file(
            geodata_path
            / "World_Administrative_Divisions/World_Administrative_Divisions.shp"
        )

        # Initialize geocoder cache for better performance
        self._geocoder_cache = {}
        self._geolocator = None
        self._cache_file = Path(self.extension_data_dir_path) / "geocoder_cache.json"

        # Initialize elevation cache for better performance
        self._elevation_cache = {}
        self._elevation_cache_file = (
            Path(self.extension_data_dir_path) / "elevation_cache.json"
        )

        # Elevation configuration
        self.enable_elevation = enable_elevation
        if self.enable_elevation:
            self.elevation_dataset = elevation_dataset
            self.elevation_endpoint = elevation_endpoint

        # Setup logger as class attribute
        self.logger = DataPipelineLogger.get_logger(
            name=self.__class__.__name__, log_file_path=self.log_file
        )

        # Load existing cache if available
        self._load_geocoder_cache()
        if self.enable_elevation:
            self._load_elevation_cache()

        # Optimize shapefiles for spatial queries by creating spatial indices
        self._optimize_shapefiles()

        os.makedirs(self.extension_data_dir_path, exist_ok=True)

    def _optimize_shapefiles(self):
        """Optimize shapefiles for efficient spatial queries."""
        try:
            # Convert to appropriate CRS for better performance (WGS84)
            # Handle cases where CRS is not set (None) by setting it first
            if self.SPATIAL_SHP.crs is None:
                self.SPATIAL_SHP = self.SPATIAL_SHP.set_crs("EPSG:4326")
            elif self.SPATIAL_SHP.crs != "EPSG:4326":
                self.SPATIAL_SHP = self.SPATIAL_SHP.to_crs("EPSG:4326")

            # Create spatial indices
            self.SPATIAL_SHP.sindex

            self.logger.info("Shapefiles optimized successfully")

        except Exception as e:
            self.logger.error(f"Error optimizing shapefiles: {e}")
            raise

    def _load_geocoder_cache(self):
        """Load previously cached geocoding results."""
        try:
            if self._cache_file.exists():
                import json

                with open(self._cache_file, "r") as f:
                    cache_data = json.load(f)
                    # Convert string keys back to tuple keys using numpy for consistency
                    self._geocoder_cache = {}
                    for k, v in cache_data.items():
                        # The save format uses f"{k[0]},{k[1]}" where k is (lat, lng)
                        # So the string format is "lat,lng"
                        lat_str, lng_str = k.split(",")
                        # Use (lat, lng) order to match existing data format
                        rounded_lat = float(np.round(float(lat_str), 3))
                        rounded_lng = float(np.round(float(lng_str), 3))
                        self._geocoder_cache[(rounded_lat, rounded_lng)] = v
                self.logger.info(
                    f"Loaded {len(self._geocoder_cache)} geocoding cache entries"
                )
        except Exception as e:
            self.logger.warning(f"Failed to load geocoder cache: {e}")
            self._geocoder_cache = {}

    def _save_geocoder_cache(self):
        """Save geocoding cache for future runs."""
        try:
            # Convert tuple keys to string keys for JSON serialization
            # Ensure consistent formatting using numpy float conversion
            cache_data = {
                f"{float(k[0])},{float(k[1])}": v
                for k, v in self._geocoder_cache.items()
            }
            with open(self._cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)
            self.logger.info(
                f"Saved {len(self._geocoder_cache)} geocoding cache entries"
            )
        except Exception as e:
            self.logger.warning(f"Failed to save geocoder cache: {e}")

    def _load_elevation_cache(self):
        """Load previously cached elevation results."""
        try:
            if self._elevation_cache_file.exists():
                with open(self._elevation_cache_file, "r") as f:
                    cache_data = json.load(f)
                    # Convert string keys back to tuple keys using numpy for consistency
                    self._elevation_cache = {}
                    for k, v in cache_data.items():
                        # The save format uses f"{k[0]},{k[1]}" where k is (lat, lng)
                        # So the string format is "lat,lng"
                        lat_str, lng_str = k.split(",")
                        # Use (lat, lng) order to match existing data format
                        rounded_lat = float(np.round(float(lat_str), 3))
                        rounded_lng = float(np.round(float(lng_str), 3))
                        self._elevation_cache[(rounded_lat, rounded_lng)] = v
                self.logger.info(
                    f"Loaded {len(self._elevation_cache)} elevation cache entries"
                )
        except Exception as e:
            self.logger.warning(f"Failed to load elevation cache: {e}")
            self._elevation_cache = {}

    def _save_elevation_cache(self):
        """Save elevation cache for future runs."""
        try:
            # Convert tuple keys to string keys for JSON serialization
            # Ensure consistent formatting using numpy float conversion
            cache_data = {
                f"{float(k[0])},{float(k[1])}": v
                for k, v in self._elevation_cache.items()
            }
            with open(self._elevation_cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)
            self.logger.info(
                f"Saved {len(self._elevation_cache)} elevation cache entries"
            )
        except Exception as e:
            self.logger.warning(f"Failed to save elevation cache: {e}")

    def run(self):
        """Main entry point for geo processing."""
        self.logger.info("Starting geo information processing...")
        self._add_geo_information()
        # Save geocoding cache for future runs
        self._save_geocoder_cache()
        self.logger.info("Geo information processing completed.")

        self._add_elevation_information()
        # Save elevation cache for future runs
        self._save_elevation_cache()

    def _add_geo_information(self):
        """
        Create geo_information table using DuckDB and parquet files.
        """

        # Initialize DuckDB connection
        conn = duckdb.connect()

        # Install and load spatial extension
        conn.execute("INSTALL spatial;")
        conn.execute("LOAD spatial;")

        try:
            # Load coordinate data from parquet file
            coordinates_file = (
                Path(self.meta_data_dir_path) / "location_coordinates.parquet"
            )
            if not coordinates_file.exists():
                self.logger.error(f"Coordinate file not found: {coordinates_file}")
                return

            self.logger.info("Loading coordinate data...")
            conn.execute(f"""
                CREATE TABLE coordinates AS 
                SELECT DISTINCT 
                    ROW_NUMBER() OVER () as location_id,
                    lat,
                    lng,
                    ST_Point(lng, lat) as location
                FROM read_parquet('{coordinates_file}')
            """)

            # Load existing geo information if available
            geo_info_file = (
                Path(self.extension_data_dir_path) / "geo_information.parquet"
            )
            existing_data = {}

            if geo_info_file.exists():
                existing_table = pq.read_table(geo_info_file)
                # Convert to Python objects for lookup
                location_col = existing_table["location"].to_pylist()
                lat_col = []
                lng_col = []
                # Extract lat and lng from WKT
                for location_wkt in location_col:
                    try:
                        # Parse "POINT(lng lat)" format
                        coords_str = location_wkt.replace("POINT(", "").replace(")", "")
                        lng_str, lat_str = coords_str.split()
                        lat_col.append(float(lat_str))
                        lng_col.append(float(lng_str))
                    except (ValueError, AttributeError) as e:
                        self.logger.warning(
                            f"Failed to parse location WKT: {location_wkt}, error: {e}"
                        )
                        # Skip invalid entries
                        continue
                # Extract other columns
                continent_col = existing_table["continent_name"].to_pylist()
                country_col = existing_table["country_name"].to_pylist()
                federal_state_col = existing_table["federal_state_name"].to_pylist()
                elevation_col = existing_table["elevation"].to_pylist()

                # Ensure all lists have the same length
                min_length = min(
                    len(lat_col),
                    len(lng_col),
                    len(continent_col),
                    len(country_col),
                    len(federal_state_col),
                    len(elevation_col),
                )

                # Convert to numpy arrays for consistent rounding
                lat_array = np.array(lat_col[:min_length])
                lng_array = np.array(lng_col[:min_length])

                # Use the same numpy rounding as in the lookup logic, convert to Python float
                rounded_lats = np.round(lat_array, 3)
                rounded_lngs = np.round(lng_array, 3)

                for i in range(min_length):
                    # Convert numpy scalars to Python floats for consistent key types
                    key = (
                        float(rounded_lngs[i]),
                        float(rounded_lats[i]),
                    )  # (lng, lat) order
                    existing_data[key] = {
                        "continent_name": continent_col[i] or "",
                        "country_name": country_col[i] or "",
                        "federal_state_name": federal_state_col[i] or "",
                        "elevation": elevation_col[i] or None,
                    }

            # Get all coordinates that need geo information using PyArrow
            coordinates_result = conn.execute("SELECT * FROM coordinates").arrow()
            coordinates_table = (
                coordinates_result.read_all()
            )  # Convert RecordBatchReader to Table

            # Extract columns as numpy arrays for vectorized processing
            location_ids_input = coordinates_table["location_id"].to_numpy()
            lats_input = coordinates_table["lat"].to_numpy()
            lngs_input = coordinates_table["lng"].to_numpy()

            self.logger.info("Loading existing geo information for caching...")

            # Group nearby coordinates to reduce API calls
            grouped_coords = self._group_nearby_coordinates(lats_input, lngs_input)
            self.logger.info(
                f"Grouped {len(lats_input)} coordinates into {len(grouped_coords)} representative points"
            )

            # Process coordinates using vectorized approach
            geo_data = self._process_coordinates_vectorized(
                location_ids_input,
                lats_input,
                lngs_input,
                existing_data,
                grouped_coords,
            )

            # Create PyArrow table and save to parquet
            geo_table = pa.table(geo_data)

            # Save geo information to parque
            pq.write_table(
                geo_table,
                self.extension_data_dir_path + "/geo_information.parquet",
                compression="BROTLI",
                max_rows_per_page=2147483647,  # Set max rows per page to avoid row group splitting
            )
            self.logger.info(f"Saved geo information to {geo_info_file}")

        except Exception as e:
            self.logger.error(f"Error processing geo information: {e}")
            raise
        finally:
            conn.close()

    def _add_elevation_information(self):
        """Add elevation information to geospatial data."""
        if not self.enable_elevation:
            self.logger.info("Elevation information processing is disabled.")
            return

        try:
            # Load existing geo information
            geo_info_file = (
                Path(self.extension_data_dir_path) / "geo_information.parquet"
            )
            if not geo_info_file.exists():
                self.logger.error(f"Geo information file not found: {geo_info_file}")
                return

            self.logger.info("Loading geo information for elevation processing...")
            geo_table = pq.read_table(geo_info_file)

            # remove existing elevation column if present
            if "elevation" in geo_table.column_names:
                geo_table = geo_table.remove_column(
                    geo_table.column_names.index("elevation")
                )

            # Extract lat and lng from location WKT
            location_col = geo_table["location"].to_pylist()
            lat_col = []
            lng_col = []
            for location_wkt in location_col:
                try:
                    # Parse "POINT(lng lat)" format
                    coords_str = location_wkt.replace("POINT(", "").replace(")", "")
                    lng_str, lat_str = coords_str.split()
                    lat_col.append(float(lat_str))
                    lng_col.append(float(lng_str))
                except (ValueError, AttributeError) as e:
                    self.logger.warning(
                        f"Failed to parse location WKT: {location_wkt}, error: {e}"
                    )
                    lat_col.append(None)
                    lng_col.append(None)

            # Prepare new elevation column
            elevations = []

            # Iterate over all coordinates and get elevation
            for lat, lon in tqdm(
                zip(lat_col, lng_col),
                total=len(lat_col),
                desc="Processing elevation data",
            ):
                if lat is None or lon is None:
                    elevations.append(None)
                    continue

                rounded_lat = float(np.round(lat, 3))
                rounded_lon = float(np.round(lon, 3))
                cache_key = (rounded_lat, rounded_lon)

                if cache_key in self._elevation_cache:
                    elevation = self._elevation_cache[cache_key]
                else:
                    elevation = self._get_elevation(rounded_lat, rounded_lon)
                    if elevation is not None:
                        self._elevation_cache[cache_key] = elevation
                    else:
                        elevation = None  # Default value if lookup fails

                elevations.append(elevation)

            # Add elevation column to geo table
            geo_table = geo_table.append_column(
                "elevation", pa.array(elevations, pa.int16())
            )

            # Save updated geo information with elevation to parquet
            pq.write_table(
                geo_table,
                self.extension_data_dir_path + "/geo_information.parquet",
                compression="BROTLI",
                max_rows_per_page=2147483647,  # Set max rows per page to avoid row group splitting
            )
            self.logger.info("Saved updated geo information with elevation to parquet.")

        except Exception as e:
            self.logger.error(f"Error processing elevation information: {e}")
            raise

    def _get_elevation(self, lat, lon):
        """Get elevation for the given coordinates using local OpenTopoData instance."""
        try:
            # Query local OpenTopoData instance
            url = f"http://{self.elevation_endpoint}/v1/{self.elevation_dataset}?locations={lon},{lat}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if "results" in data and len(data["results"]) > 0:
                    elevation = data["results"][0].get("elevation", 0)
                    if elevation is not None:
                        return int(elevation)
            return None
        except Exception as e:
            self.logger.warning(f"Elevation lookup failed for {lat}, {lon}: {e}")
            return None

    def _group_nearby_coordinates(self, lats, lngs, distance_threshold=0.01):
        """Group coordinates that are within threshold distance to reduce API calls."""
        grouped_coords = defaultdict(list)

        # Convert to rounded coordinates for grouping
        rounded_lats = np.round(lats, 2)  # Round to ~1km precision
        rounded_lngs = np.round(lngs, 2)

        # Group coordinates by rounded values
        for i in range(len(lats)):
            key = (rounded_lats[i], rounded_lngs[i])
            grouped_coords[key].append(i)

        return dict(grouped_coords)

    def _create_geocoding_groups(self, new_indices, lats, lngs, grouped_coords):
        """Create groups of coordinates that should share the same geocoding result."""
        geocoding_groups = defaultdict(list)

        # Process each coordinate group to determine which need geocoding
        for group_key, group_indices in grouped_coords.items():
            # Find indices that need geocoding (are in new_indices)
            indices_needing_geocoding = [
                idx for idx in group_indices if idx in new_indices
            ]

            if indices_needing_geocoding:
                # Use the first coordinate as representative for this group
                repr_idx = indices_needing_geocoding[0]
                geocoding_groups[group_key] = indices_needing_geocoding

        return dict(geocoding_groups)

    def _batch_geocode_groups(self, geocoding_groups):
        """Perform batch geocoding and spatial lookups for representative coordinates from each group."""
        group_results = {}

        # Extract all representative coordinates for batch processing
        coords_list = list(geocoding_groups.keys())

        if not coords_list:
            return group_results

        # Convert to numpy arrays for vectorized operations
        lats = np.array([coord[0] for coord in coords_list])
        lngs = np.array([coord[1] for coord in coords_list])

        # Perform batch spatial lookups
        continents = self._batch_spatial_lookup(
            lats, lngs, self.SPATIAL_SHP, "CONTINENT"
        )
        countries = self._batch_spatial_lookup(lats, lngs, self.SPATIAL_SHP, "ISO_CC")
        federal_states = self._batch_spatial_lookup(
            lats, lngs, self.SPATIAL_SHP, "ISO_SUB"
        )
        # federal states are in format "ISO_CC-ISO_SUB", we want to combine country and federal state for better caching
        federal_states = [
            f"{countries[i]}-{federal_states[i]}" if federal_states[i] else ""
            for i in range(len(federal_states))
        ]

        self.logger.info(
            f"Completed batch spatial lookups for {len(coords_list)} coordinate groups"
        )

        for i, group_key in enumerate(
            tqdm(coords_list, desc="Geocoding representative coordinates")
        ):
            # Combine all results
            group_results[group_key] = {
                "continent_name": continents[i],
                "country_name": countries[i],
                "federal_state_name": federal_states[i],
                "elevation": None,  # Placeholder, will be filled later
            }

        return group_results

    def _batch_spatial_lookup(
        self, lats, lngs, shapefile, name_column, is_numeric=False
    ):
        """Perform batch spatial lookup for multiple coordinates."""
        results = []

        try:
            # Create points for all coordinates
            points = [Point(lng, lat) for lat, lng in zip(lats, lngs)]

            for point in tqdm(
                points,
                desc=f"Batch spatial lookup for {name_column}",
                total=len(points),
            ):
                try:
                    # Use spatial index for faster querying
                    possible_matches_index = list(
                        shapefile.sindex.intersection(point.bounds)
                    )
                    possible_matches = shapefile.iloc[possible_matches_index]

                    # Check actual containment
                    precise_matches = possible_matches[possible_matches.contains(point)]

                    if not precise_matches.empty:
                        value = precise_matches.iloc[0][name_column]
                        if is_numeric:
                            try:
                                results.append(
                                    int(value) if value and str(value).isdigit() else 0
                                )
                            except (ValueError, TypeError):
                                results.append(0)
                        else:
                            results.append(str(value) if value else "")
                    else:
                        results.append(0 if is_numeric else "")

                except Exception as e:
                    self.logger.warning(f"Spatial lookup failed for point {point}: {e}")
                    results.append(0 if is_numeric else "")

            return results

        except Exception as e:
            self.logger.error(f"Batch spatial lookup failed: {e}")
            return [0 if is_numeric else "" for _ in range(len(lats))]

    def _process_coordinates_vectorized(
        self, location_ids, lats, lngs, existing_data, grouped_coords
    ):
        """Process coordinates using vectorized operations."""

        # Initialize output arrays
        n_coords = len(location_ids)
        location_ids_out = location_ids.copy()
        locations_out = [f"POINT({lng} {lat})" for lat, lng in zip(lngs, lats)]

        # Initialize with default values using numpy
        continent_names = np.full(n_coords, "", dtype=object)
        country_names = np.full(n_coords, "", dtype=object)
        federal_state_names = np.full(n_coords, "", dtype=object)
        elevations = np.full(n_coords, None, dtype=object)

        # Create vectorized lookup for existing data
        rounded_lats = np.round(lats, 3)
        rounded_lngs = np.round(lngs, 3)
        # Convert to Python floats for consistent key types
        # Use (lat, lng) order to match existing data format
        cache_keys = [
            (float(lat), float(lng)) for lat, lng in zip(rounded_lats, rounded_lngs)
        ]

        cached_mask = np.array([key in existing_data for key in cache_keys])
        new_mask = ~cached_mask

        # Process cached coordinates vectorized
        cached_indices = np.where(cached_mask)[0]
        for idx in cached_indices:
            key = cache_keys[idx]
            cached_data = existing_data[key]
            continent_names[idx] = cached_data["continent_name"]
            country_names[idx] = cached_data["country_name"]
            federal_state_names[idx] = cached_data["federal_state_name"]
            elevations[idx] = cached_data["elevation"]

        # Process new coordinates by groups more efficiently
        new_indices = np.where(new_mask)[0]

        # Group new coordinates by geographic proximity for batch processing
        geocoding_groups = self._create_geocoding_groups(
            new_indices, lats, lngs, grouped_coords
        )

        self.logger.info(
            f"Processing {len(cached_indices)} cached and {len(new_indices)} new coordinates in {len(geocoding_groups)} geocoding groups"
        )

        # Additional debug logging
        if len(cached_indices) > 0:
            self.logger.info(
                f"Successfully found {len(cached_indices)} coordinates in existing cache"
            )
        if len(new_indices) > 0:
            self.logger.info(f"Need to process {len(new_indices)} new coordinates")

        # Process representative coordinates for each group
        group_results = self._batch_geocode_groups(geocoding_groups)

        # Apply results to all coordinates in each group
        for group_key, group_indices in tqdm(
            geocoding_groups.items(), desc="Applying geocoding results"
        ):
            geo_info = group_results[group_key]

            for idx in group_indices:
                continent_names[idx] = geo_info["continent_name"]
                country_names[idx] = geo_info["country_name"]
                federal_state_names[idx] = geo_info["federal_state_name"]
                elevations[idx] = geo_info["elevation"]

        return {
            "location_id": location_ids_out.tolist(),
            "location": locations_out,
            "continent_name": continent_names.tolist(),
            "country_name": country_names.tolist(),
            "federal_state_name": federal_state_names.tolist(),
            "elevation": elevations.tolist(),
        }
