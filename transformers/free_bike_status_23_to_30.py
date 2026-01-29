import os
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
import tqdm

from utils.data_pipeline_logger import DataPipelineLogger


class FreeBikeStatusTransformer:
    def __init__(
        self, input_data_dir_path, export_data_dir_path, logs_data_dir_path, operator
    ):
        if not hasattr(self, "initialized"):
            self.initialized = True
            self.operator = operator
            self.input_data_dir_path = input_data_dir_path
            self.export_data_dir_path = export_data_dir_path
            self.logs_data_dir_path = logs_data_dir_path
            self.log_file = Path(logs_data_dir_path) / "logs.log"

            # Setup logger
            self.logger = DataPipelineLogger.get_logger(
                name=self.__class__.__name__, log_file_path=self.log_file
            )

    def run_transformer(self):
        # if output directory doesn't exist, create it
        os.makedirs(self.export_data_dir_path, exist_ok=True)
        self.logger.info(
            f"Starting FreeBikeStatus transformation for operator {self.operator}."
        )
        ndate = len(os.listdir(self.input_data_dir_path))
        for date in tqdm.tqdm(
            os.listdir(self.input_data_dir_path), desc="Processing dates", total=ndate
        ):
            # open parquet file in input_dir
            try:
                table = pq.read_table(
                    os.path.join(
                        self.input_data_dir_path, date, "free_bike_status.parquet"
                    )
                )
            except Exception as e:
                self.logger.error(
                    f"Error reading parquet file {os.path.join(self.input_data_dir_path, date, 'free_bike_status.parquet')}: {e} for operator {self.operator}."
                )
                return None

            if not "bike_id" in table.column_names:
                self.logger.error(
                    f"Input table does not have 'bike_id' column for operator {self.operator}."
                )
                return None
            if not "last_reported" in table.column_names:
                self.logger.error(
                    f"Input table does not have 'last_reported' column for operator {self.operator}."
                )
                return None
            if not "last_updated" in table.column_names:
                self.logger.error(
                    f"Input table does not have 'last_updated' column for operator {self.operator}."
                )
                return None

            # Rename columns
            table = table.rename_columns({"bike_id": "vehicle_id"})

            # Convert last_reported (already in UTC seconds) to timestamp[ns, tz=UTC] not null
            table = table.set_column(
                table.schema.get_field_index("last_reported"),
                "last_reported",
                pc.cast(
                    pc.multiply(
                        pc.cast(table["last_reported"], pa.int64()), 1_000_000_000
                    ),
                    pa.timestamp("ns", tz="UTC"),
                ),
            ).set_column(
                table.schema.get_field_index("last_updated"),
                pa.field("last_updated", pa.timestamp("ns", tz="UTC"), nullable=False),
                pc.cast(
                    pc.multiply(
                        pc.cast(table["last_updated"], pa.int64()), 1_000_000_000
                    ),
                    pa.timestamp("ns", tz="UTC"),
                ),
            )

            # overwrite version column to 3.0
            if "version" in table.column_names:
                table = table.set_column(
                    table.schema.get_field_index("version"),
                    pa.field("version", pa.string(), nullable=False),
                    pa.array(["3.0"] * len(table), type=pa.string()),
                )

            # export to parquet in output_dir
            output_path = os.path.join(
                self.export_data_dir_path, date, "vehicle_status.parquet"
            )

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            try:
                pq.write_table(table, output_path, compression="BROTLI")
            except Exception as e:
                self.logger.error(
                    f"Error writing parquet file {output_path}: {e} for operator {self.operator}."
                )
                return None
