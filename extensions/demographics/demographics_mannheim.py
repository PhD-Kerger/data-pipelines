from pathlib import Path
import json
import pyarrow as pa
import pyarrow.parquet as pq
import datetime

from utils import DataPipelineLogger


class Demographics_MA:
    def __init__(
        self,
        extension_data_dir_path,
        meta_data_dir_path,
        input_data_dir_path,
        logs_data_dir_path,
        file,
    ):
        self.extension_data_dir_path = extension_data_dir_path + "/demographics_MA"
        self.meta_data_dir_path = meta_data_dir_path
        self.log_file = Path(logs_data_dir_path) / "logs.log"
        self.file = Path(file)

        # Setup logger as class attribute
        self.logger = DataPipelineLogger.get_logger(
            name=self.__class__.__name__, log_file_path=self.log_file
        )

        if not Path(self.extension_data_dir_path).exists():
            Path(self.extension_data_dir_path).mkdir(parents=True, exist_ok=True)

    def load_data(self, input_file):
        """Load demographics data from a JSON file."""
        with open(input_file, "r") as f:
            data = json.load(f)
        return data["geographies"][1]

    def run(self):
        self.data = self.load_data(self.file)
        self.logger.info(f"Loaded demographics data from JSON file {self.file}")
        self.features = self.getFeatures(self.data)
        self.logger.info(
            f"Extracted {len(self.features)} features from demographics data"
        )
        self.themes = self.getThemes(self.data, self.features)
        self.logger.info(f"Extracted {len(self.themes)} themes from demographics data")
        self.jsonToParquet(self.themes)
        self.logger.info("Converted demographics data to parquet format")

    def getThemes(self, data, features):
        themes = {}
        for theme in data["themes"]:
            theme_name = theme["name"]
            indicators = []
            for indicator in theme["indicators"]:
                if indicator["date"] not in ["2023", "2024", "2025"]:
                    continue
                for value, feature in zip(indicator["values"], features):
                    indicators.append(
                        {
                            "city": "MA",
                            "area": feature,
                            "feature": indicator["name"],
                            "year": indicator["date"],
                            "value": value,
                        }
                    )
            themes[theme_name] = indicators
        return themes

    def getFeatures(self, data):
        return [feature["name"] for feature in data["features"]]

    def jsonToParquet(self, data):
        """
        Creates a single parquet file with all demographic data.
        Adds a 'topic' column to identify the main theme.
        """
        all_records = []

        for theme, records in data.items():
            for record in records:
                # Convert value to appropriate type
                value = record["value"]
                if isinstance(value, str):
                    try:
                        # Try to convert string to float
                        value = float(value) if value.strip() else None
                    except (ValueError, AttributeError):
                        # Keep as string if conversion fails
                        pass

                # Add the topic column to each record
                record_with_topic = {
                    "topic": theme,
                    "city": record["city"],
                    "area": record["area"],
                    "feature": record["feature"],
                    # year as yyyy-mm-dd format for parquet compatibility. Use first day of year.
                    "year": datetime.datetime(
                        int(record["year"]), 1, 1, tzinfo=datetime.timezone.utc
                    ),
                    "value": value,
                }
                all_records.append(record_with_topic)

        # Create single table with all data
        schema = pa.schema(
            [
                ("topic", pa.string()),
                ("city", pa.string()),
                ("area", pa.string()),
                ("feature", pa.string()),
                ("year", pa.timestamp("us", tz="UTC")),
                ("value", pa.float64()),
            ]
        )

        table = pa.Table.from_pylist(all_records, schema=schema)
        pq.write_table(
            table, self.extension_data_dir_path + "/demographics_mannheim.parquet", compression="BROTLI"
        )
