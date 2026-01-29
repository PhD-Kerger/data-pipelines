import os
from pathlib import Path
import osmnx as ox
import pyarrow as pa
import pyarrow.parquet as pq
import geoarrow.pyarrow as ga

from utils.data_pipeline_logger import DataPipelineLogger


class OSMLanduse:
    def __init__(
        self,
        extension_data_dir_path,
        meta_data_dir_path,
        logs_data_dir_path,
        input_data_dir_path,
        cities,
    ):
        self.extension_data_dir_path = extension_data_dir_path + "/osm_landuse"
        self.meta_data_dir_path = meta_data_dir_path
        self.log_file = Path(logs_data_dir_path) / "logs.log"
        self.cities = cities if cities else []
        self.tags = {"landuse": True}

        # Setup logger
        self.logger = DataPipelineLogger.get_logger(
            name=self.__class__.__name__, log_file_path=self.log_file
        )

        os.makedirs(self.extension_data_dir_path, exist_ok=True)

    def run(self):
        """Main method to run OSM data processing"""
        if len(self.cities) == 0:
            self.logger.warning(
                "No cities provided for OSM landuse extension. Terminating."
            )
            return

        # Step 1: Get area IDs for cities
        landuse_data = self._get_landuse()

        # Step 2: Save landuse data as Parquet files
        self._save_landuse_data(landuse_data)

        self.logger.info("OSM processing completed")

    def _get_landuse(self):
        """Fetch landuse data from OSM for the specified cities"""
        landuse_data = {}
        for city in self.cities:
            self.logger.info(f"Processing landuse data for city: {city}")
            gdf = ox.features_from_place(city, self.tags)

            # Keep only polygons
            gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]

            # Keep relevant columns
            gdf = gdf[["landuse", "geometry"]]

            landuse_data[city] = gdf

            self.logger.info(f"Fetched landuse data for {city} with tags: {self.tags}")
        return landuse_data

    def _save_landuse_data(self, landuse_data):
        """Save landuse GeoDataFrames as a single Parquet file"""
        output_file = os.path.join(
            self.extension_data_dir_path, "osm_landuse.parquet"
        )

        # Collect all data from all cities
        all_ids = []
        all_cities = []
        all_landuses = []
        all_geometries = []
        
        current_id = 1
        for city, gdf in landuse_data.items():
            # Reset index to avoid saving element/id as index columns
            gdf = gdf.reset_index(drop=True)
            
            num_rows = len(gdf)
            
            # Add data for this city
            all_ids.extend(range(current_id, current_id + num_rows))
            all_cities.extend([city] * num_rows)
            all_landuses.extend(gdf["landuse"].tolist())
            all_geometries.extend(
                gdf["geometry"]
                .apply(lambda geom: geom.wkb if geom is not None else None)
                .tolist()
            )
            
            current_id += num_rows
            self.logger.info(f"Added {num_rows} landuse features for {city}")

        # Prepare combined data dictionary with WKB geometry
        formatted_data = {
            "id": all_ids,
            "city": all_cities,
            "landuse": all_landuses,
            "area": all_geometries,
        }

        # Define schema with geoarrow WKB type for PostGIS compatibility
        schema = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("city", pa.string()),
            pa.field("landuse", pa.string()),
            pa.field("area", ga.wkb()),
        ])

        # Create PyArrow table with proper schema
        table = pa.Table.from_pydict(formatted_data, schema=schema)

        # Write to Parquet file
        pq.write_table(table, output_file, compression="BROTLI")

        self.logger.info(f"Saved landuse data for all cities ({len(all_ids)} total features) to {output_file}")
