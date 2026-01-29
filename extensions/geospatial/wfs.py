from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from owslib.wfs import WebFeatureService
import geopandas as gpd
import urllib3
import ssl
import geoarrow.pyarrow as ga

from utils import DataPipelineLogger

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context


class WFS:
    def __init__(
        self,
        extension_data_dir_path,
        meta_data_dir_path,
        input_data_dir_path,
        logs_data_dir_path,
        wfs_name,
        wfs_url,
        wfs_version,
        wfs_layer_name,
    ):
        self.extension_data_dir_path = extension_data_dir_path + "/wfs"
        self.meta_data_dir_path = meta_data_dir_path
        self.log_file = Path(logs_data_dir_path) / "logs.log"

        self.wfs_name = wfs_name
        self.wfs_url = wfs_url
        self.wfs_version = wfs_version
        self.wfs_layer_name = wfs_layer_name

        # Setup logger as class attribute
        self.logger = DataPipelineLogger.get_logger(
            name=self.__class__.__name__, log_file_path=self.log_file
        )

        if not Path(self.extension_data_dir_path).exists():
            Path(self.extension_data_dir_path).mkdir(parents=True, exist_ok=True)

        try:
            self.wfs = WebFeatureService(wfs_url, version=self.wfs_version)
        except Exception as e:
            self.logger.error(f"Error initializing WFS: {e}")
            self.wfs = None

    def run(self):
        layers = self.getLayers()
        self.logger.info(f"Available WFS layers for endpoint: {layers}")
        response = self.getWFSFeature()
        if response:
            gdf = gpd.read_file(response)
            gdf = gdf.to_crs(epsg=4326)
            gdf.to_file(
                self.extension_data_dir_path + "/" + self.wfs_name + ".geojson",
                driver="GeoJSON",
            )
            self.logger.info(
                f"Saved {len(gdf)} features to {self.extension_data_dir_path + '/' + self.wfs_name + '.geojson'}"
            )
        else:
            self.logger.error("No response received for WFS feature request.")
            return
        self.formatResponse(gdf)

    def getLayers(self):
        return list(self.wfs.contents.keys())

    def getWFSFeature(self):
        try:
            response = self.wfs.getfeature(
                typename=[self.wfs_layer_name],  # Correct layer name with namespace
            )
            self.logger.info(
                f"Successfully retrieved WFS feature for layer: {self.wfs_layer_name}"
            )
        except Exception as e:
            self.logger.error(f"Error retrieving WFS feature: {e}")
            response = None
        return response

    def formatResponse(self, gdf):
        # Format WFS response from GeoJSON to Parquet with city specific schemas
        if self.wfs_name == "Mannheim":
            num_rows = len(gdf)
            filtered_data = {
                "city": ["MA"] * num_rows,
                "name": [gdf["id_name"][i] for i in range(num_rows)],
                "area": gdf["geometry"]
                .apply(lambda geom: geom.wkb if geom is not None else None)
                .tolist(),
            }
        else:
            self.logger.warning(
                f"No formatting rules defined for WFS name: {self.wfs_name}"
            )
            return
        schema = pa.schema(
            [
                pa.field("city", pa.string()),
                pa.field("name", pa.string()),
                pa.field("area", ga.wkb()),
            ]
        )

        table = pa.Table.from_pydict(filtered_data, schema=schema)

        pq.write_table(
            table,
            self.extension_data_dir_path + "/" + self.wfs_name + ".parquet", compression="BROTLI"
        )
        self.logger.info(
            f"Saved formatted Parquet file to {self.extension_data_dir_path + '/' + self.wfs_name + '.parquet'}"
        )
