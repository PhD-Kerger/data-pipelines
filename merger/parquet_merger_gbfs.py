import pyarrow_ops as pa_ops
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
from pathlib import Path
from datetime import datetime, timezone
import re

from utils.data_pipeline_logger import DataPipelineLogger


class ParquetMergerGBFS:
    def __init__(self, log_file_path):
        if not hasattr(self, "initialized"):
            self.initialized = True
            # Setup logger
            self.logger = DataPipelineLogger.get_logger(
                name=self.__class__.__name__, log_file_path=log_file_path
            )

    def read_parquet_file(self, file_path):
        """Read a parquet file and return as PyArrow Table"""
        try:
            table = pq.read_table(file_path)
            return table
        except Exception as e:
            self.logger.error(f"Error reading parquet file {file_path}: {e}")
            return None

    def get_date_folders(self, base_path):
        """Get all date folders (yyyy-mm-dd format) from a base path"""
        date_folders = []
        base_path = Path(base_path)

        if not base_path.exists():
            self.logger.warning(f"Path {base_path} does not exist")
            return date_folders

        for item in base_path.iterdir():
            if item.is_dir():
                # Check if folder name matches yyyy-mm-dd format
                try:
                    datetime.strptime(item.name, "%Y-%m-%d")
                    date_folders.append(item.name)
                except ValueError:
                    continue

        return sorted(date_folders)

    def find_parquet_files(self, folder_path):
        """Find all parquet files in a folder"""
        folder_path = Path(folder_path)
        parquet_files = list(folder_path.glob("*.parquet"))
        return parquet_files

    def compare_and_merge_parquet_files(
        self, table_a, table_b, timestamp_column="timestamp", operator=None
    ):
        """
        Compare two parquet files and return merged Table with unique records from both files

        Args:
            table_a: PyArrow Table from parquet file A
            table_b: PyArrow Table from parquet file B
            timestamp_column: Name of the timestamp column to compare on

        Returns:
            pyarrow.Table: Merged Table with unique records from both files
        """

        if table_a.num_rows == 0 and table_b.num_rows == 0:
            self.logger.warning(f"Both input tables are empty for operator {operator}")
            return None
        if table_a.num_rows == 0:
            self.logger.info(
                f"Table A is empty for operator {operator}, returning Table B"
            )
            table_b = pa_ops.drop_duplicates(
                table_b, [timestamp_column, "vehicle_id"], keep="first"
            )
            return table_b
        if table_b.num_rows == 0:
            self.logger.info(
                f"Table B is empty for operator {operator}, returning Table A"
            )
            table_a = pa_ops.drop_duplicates(
                table_a, [timestamp_column, "vehicle_id"], keep="first"
            )
            return table_a

        # make distinct on timestamp column and vehicle_id if exists
        if (
            "vehicle_id" in table_a.column_names
            and "vehicle_id" in table_b.column_names
        ):
            table_a = pa_ops.drop_duplicates(
                table_a, [timestamp_column, "vehicle_id"], keep="first"
            )
            table_b = pa_ops.drop_duplicates(
                table_b, [timestamp_column, "vehicle_id"], keep="first"
            )
        else:
            self.logger.error(
                f"Input tables do not have 'vehicle_id' column for operator {operator}"
            )
            return None

        # Get unique timestamps from each table
        timestamps_a = pc.unique(table_a[timestamp_column])
        timestamps_b = pc.unique(table_b[timestamp_column])

        # Find unique timestamps in A (not in B)
        unique_in_a_mask = pc.invert(pc.is_in(table_a[timestamp_column], timestamps_b))
        unique_in_b_mask = pc.invert(pc.is_in(table_b[timestamp_column], timestamps_a))

        # Filter records with unique timestamps
        records_unique_in_a = table_a.filter(unique_in_a_mask)
        records_unique_in_b = table_b.filter(unique_in_b_mask)
        records_common = table_a.filter(
            pc.is_in(table_a[timestamp_column], timestamps_b)
        )

        unique_count_a = len(records_unique_in_a)
        unique_count_b = len(records_unique_in_b)
        common_count = len(records_common)

        self.logger.info(f"Common records: {common_count} for operator {operator}")
        self.logger.info(
            f"Unique records in A: {unique_count_a} for operator {operator}"
        )
        self.logger.info(
            f"Unique records in B: {unique_count_b} for operator {operator}"
        )

        # If no unique records, return common table
        if unique_count_a == 0 and unique_count_b == 0:
            self.logger.info(
                f"No unique records found in either file for operator {operator}"
            )
            # Sort by timestamp
            table_a_sorted = pc.sort_indices(table_a[timestamp_column])
            sorted_table = pc.take(table_a, table_a_sorted)
            return sorted_table

        # Concatenate the unique records
        if unique_count_a > 0 and unique_count_b > 0:
            merged_table = pa.concat_tables([records_unique_in_a, records_unique_in_b])
        elif unique_count_a > 0:
            merged_table = records_unique_in_a
        else:
            merged_table = records_unique_in_b

        # add the common records
        if common_count > 0:
            merged_table = pa.concat_tables([merged_table, records_common])

        # Sort by timestamp
        sort_indices = pc.sort_indices(merged_table[timestamp_column])
        merged_table = pc.take(merged_table, sort_indices)

        # merged table need to have at least same amount of rows as the larger of the two unique counts
        if len(merged_table) < max(unique_count_a, unique_count_b):
            self.logger.error(
                f"Merged table has fewer records than expected for operator {operator}"
            )
            return None

        if len(merged_table) != (unique_count_a + unique_count_b + len(records_common)):
            self.logger.error(
                f"Merged table row count does not match sum of unique and common records for operator {operator}"
            )
            return None

        self.logger.info(
            f"Merged {len(merged_table)} unique records for operator {operator}"
        )

        return merged_table

    def update_ttl_to_uint64(self, table):
        table = table.set_column(
            table.schema.get_field_index("ttl"),
            pa.field("ttl", pa.uint64(), nullable=False),
            pc.cast(pc.cast(table["ttl"], pa.int64()), pa.uint64()),
        )
        return table

    def process_date_comparison(
        self,
        folder_a,
        folder_b,
        output_folder,
        date_str,
        timestamp_column="timestamp",
        operator=None,
    ):
        """
        Process comparison for a specific date between folder A and B

        Args:
            folder_a: Base path to folder A
            folder_b: Base path to folder B
            output_folder: Base path to output folder
            date_str: Date string in yyyy-mm-dd format
            timestamp_column: Name of the timestamp column
        """
        date_folder_a = Path(folder_a) / date_str
        date_folder_b = Path(folder_b) / date_str
        output_date_folder = Path(output_folder) / date_str

        if not date_folder_a.exists():
            self.logger.error(
                f"Date folder {date_folder_a} does not exist in folder A for operator {operator}"
            )
            return

        if not date_folder_b.exists():
            self.logger.error(
                f"Date folder {date_folder_b} does not exist in folder B for operator {operator}"
            )
            return

        # Create output directory
        output_date_folder.mkdir(parents=True, exist_ok=True)

        # Find parquet files in both folders
        files_a = self.find_parquet_files(date_folder_a)
        files_b = self.find_parquet_files(date_folder_b)

        if not files_a:
            self.logger.warning(
                f"No parquet files found in {date_folder_a} for operator {operator}"
            )
            return

        if not files_b:
            self.logger.warning(
                f"No parquet files found in {date_folder_b} for operator {operator}"
            )
            return

        if (len(files_a) != len(files_b)) and len(files_a) != 1:
            self.logger.error(
                f"Number of parquet files in A ({len(files_a)}) and B ({len(files_b)}) do not match for date {date_str} and operator {operator}"
            )
            return

        file_a = files_a[0]
        file_b = files_b[0]

        self.logger.info(
            f"Processing files for date {date_str} for operator {operator}:"
        )

        if (
            file_a.name != "vehicle_status.parquet"
            or file_b.name != "vehicle_status.parquet"
        ):
            self.logger.error(
                f"Parquet files must be named 'vehicle_status.parquet'. Found: {file_a.name}, {file_b.name} for operator {operator}"
            )
            return

        # Read parquet files
        table_a = self.read_parquet_file(file_a)
        table_b = self.read_parquet_file(file_b)

        # if file is v3.0, ensure last_reported is correct type
        if re.match(r"^vehicle_status\.parquet$", file_a.name):
            self.logger.info(
                f"Ensuring file A has correct GBFS 3.0 schema for operator {operator}"
            )
            table_a = self.change_last_reported_in_30_schema(table_a, operator=operator)
            table_a = self.update_ttl_to_uint64(table_a)
        if re.match(r"^vehicle_status\.parquet$", file_b.name):
            self.logger.info(
                f"Ensuring file B has correct GBFS 3.0 schema for operator {operator}"
            )
            table_b = self.change_last_reported_in_30_schema(table_b, operator=operator)
            table_b = self.update_ttl_to_uint64(table_b)

        # check if schemas match
        if table_a.schema != table_b.schema:
            self.logger.error(
                f"Schemas of the two parquet files do not match after conversion to GBFS 3.0 for operator {operator}:"
            )
            self.logger.error(f"Schema A: {table_a.schema}")
            self.logger.error(f"Schema B: {table_b.schema}")
            return

        merged_table = self.compare_and_merge_parquet_files(
            table_a, table_b, timestamp_column, operator=operator
        )

        if merged_table is None:
            self.logger.error(f"Merging parquet files failed for operator {operator}")
            return

        # if any timestamp in the field timestamp, last_updated or last_reported is before 2000-01-01, log error
        for ts_col in ["last_updated", "last_reported"]:
            if ts_col in merged_table.column_names:
                min_timestamp = pc.min(merged_table[ts_col]).as_py()
                if ts_col == "last_reported":
                    # get unique values
                    unique_values = pc.unique(merged_table[ts_col])
                    if (
                        len(unique_values) == 1
                        and str(unique_values[0].as_py())
                        == "1754-08-30 22:43:41.128654848+00:00"
                    ):
                        self.logger.warning(
                            f"All values in 'last_reported' column are error metric from Go for operator {operator}, setting column to NaT"
                        )
                        merged_table = merged_table.set_column(
                            merged_table.schema.get_field_index("last_reported"),
                            "last_reported",
                            pa.array(
                                [None] * merged_table.num_rows,
                                type=pa.timestamp("ns", tz="UTC"),
                            ),
                        )
                        min_timestamp = None

                if min_timestamp != None and min_timestamp < datetime(
                    2023, 1, 1, tzinfo=timezone.utc
                ):
                    self.logger.warning(
                        f"Timestamp column '{ts_col}' has values before 2023-01-01: {min_timestamp} for operator {operator}"
                    )
                    merged_table = merged_table.filter(
                        pc.greater_equal(
                            merged_table[ts_col],
                            datetime(2023, 1, 1, tzinfo=timezone.utc),
                        )
                    )
        # export merged table
        output_file_path = output_date_folder / "vehicle_status.parquet"
        pq.write_table(merged_table, output_file_path, compression="BROTLI")
        self.logger.info(
            f"Merged parquet file written to {output_file_path} for operator {operator}"
        )

    def change_last_reported_in_30_schema(self, table, operator=None):
        """
        Change the last_reported column in a PyArrow Table with GBFS 3.0 schema to ensure correct type

        Args:
            table: PyArrow Table with GBFS 3.0 schema
        """
        if not "last_reported" in table.column_names:
            self.logger.error(
                f"Input table does not have 'last_reported' column for operator {operator}"
            )
            return None

        # Convert last_reported (already in UTC seconds) to timestamp[ns, tz=UTC] not null
        table = table.set_column(
            table.schema.get_field_index("last_reported"),
            "last_reported",
            pc.cast(table["last_reported"], pa.timestamp("ns", tz="UTC")),
        )
        return table

    def merge_parquet_files_by_date(
        self,
        folder_a,
        folder_b,
        output_folder,
        timestamp_column="last_updated",
        operator=None,
    ):
        """
        Main function to compare and merge parquet files between two folders by date

        Args:
            folder_a: Path to folder A (/comp/A)
            folder_b: Path to folder B (/comp/B)
            output_folder: Path to output folder (/comp/merge)
            timestamp_column: Name of the timestamp column to compare on
        """
        self.logger.info(
            f"Starting parquet file comparison and merge process for operator {operator}"
        )
        self.logger.info(f"Folder A: {folder_a} for operator {operator}")
        self.logger.info(f"Folder B: {folder_b} for operator {operator}")
        self.logger.info(f"Output folder: {output_folder} for operator {operator}")

        # Get date folders from both directories
        dates_a = set(self.get_date_folders(folder_a))
        dates_b = set(self.get_date_folders(folder_b))

        # One list with all dates
        all_dates = dates_a | dates_b

        # Process each common date
        for date_str in sorted(all_dates):
            self.logger.info(f"Processing date: {date_str} for operator {operator}")
            # if date is only in one folder, use the same files
            src1 = Path(folder_a)
            src2 = Path(folder_b)
            self.logger.info(f"  Source A: {src1} for operator {operator}")
            self.logger.info(f"  Source B: {src2} for operator {operator}")

            if date_str not in dates_a:
                src1 = Path(folder_b)
            if date_str not in dates_b:
                src2 = Path(folder_a)

            self.process_date_comparison(
                src1, src2, output_folder, date_str, timestamp_column, operator
            )

        self.logger.info(
            f"Parquet file comparison and merge process completed for operator {operator}"
        )
