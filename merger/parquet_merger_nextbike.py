import pyarrow_ops as pa_ops
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
from pathlib import Path
from datetime import datetime

from utils.data_pipeline_logger import DataPipelineLogger


class ParquetMergerNextbike:
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
        self, table_a, table_b, id_column, timestamp_column="timestamp", operator=None
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
                table_b, [timestamp_column, id_column], keep="first"
            )
            return table_b
        if table_b.num_rows == 0:
            self.logger.info(
                f"Table B is empty for operator {operator}, returning Table A"
            )
            table_a = pa_ops.drop_duplicates(
                table_a, [timestamp_column, id_column], keep="first"
            )
            return table_a

        # make distinct on timestamp column and vehicle_id if exists
        if id_column in table_a.column_names and id_column in table_b.column_names:
            table_a = pa_ops.drop_duplicates(
                table_a, [timestamp_column, id_column], keep="first"
            )
            table_b = pa_ops.drop_duplicates(
                table_b, [timestamp_column, id_column], keep="first"
            )
        else:
            self.logger.error(
                f"Input tables do not have {id_column} column for operator {operator}"
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
            return table_a

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

        # Ensure files are sorted for consistent pairing
        files_a.sort()
        files_b.sort()

        if (len(files_a) != len(files_b)) and len(files_a) != 4:
            self.logger.error(
                f"Number of parquet files in A ({len(files_a)}) and B ({len(files_b)}) do not match for date {date_str} and operator {operator}"
            )
            return

        bikes_file = "Bikes.parquet"
        cities_file = "Cities.parquet"
        countries_file = "Countries.parquet"
        places_file = "Places.parquet"

        self.logger.info(
            f"Processing files for date {date_str} for operator {operator}:"
        )

        for i in range(len(files_a)):
            file_a = files_a[i]
            file_b = files_b[i]

            self.logger.info(
                f"Comparing {file_a.name} and {file_b.name} for operator {operator}"
            )

            if file_a.name == bikes_file and file_b.name == bikes_file:
                id_column = "number"
                filename = bikes_file
            elif file_a.name == cities_file and file_b.name == cities_file:
                id_column = "uid"
                filename = cities_file
            elif file_a.name == countries_file and file_b.name == countries_file:
                id_column = "name"
                filename = countries_file
            elif file_a.name == places_file and file_b.name == places_file:
                id_column = "uid"
                filename = places_file
            else:
                self.logger.error(
                    f"Unrecognized parquet file names: {file_a.name}, {file_b.name} for operator {operator}"
                )
                return

            # Read parquet files
            table_a = self.read_parquet_file(file_a)
            table_b = self.read_parquet_file(file_b)

            # check if schemas match
            if table_a.schema != table_b.schema:
                self.logger.error(
                    f"Schemas of the two parquet files {file_a.name} and {file_b.name} do not match for operator {operator}:"
                )
                self.logger.error(f"Schema A: {table_a.schema}")
                self.logger.error(f"Schema B: {table_b.schema}")
                return

            self.logger.info(
                f"Merging {file_a.name} and {file_b.name} with ID column '{id_column}' for operator {operator}"
            )

            merged_table = self.compare_and_merge_parquet_files(
                table_a, table_b, timestamp_column, id_column, operator=operator
            )

            if merged_table is None:
                self.logger.error(
                    f"Merging parquet files {file_a.name} and {file_b.name} failed"
                )
                return

            # export merged table
            output_file_path = output_date_folder / filename
            # sort table by id_column and timestamp before writing
            sort_indices = pc.sort_indices(
                merged_table,
                [(id_column, "ascending"), (timestamp_column, "ascending")],
            )
            merged_table = pc.take(merged_table, sort_indices)

            pq.write_table(merged_table, output_file_path, compression="BROTLI")
            self.logger.info(
                f"Merged parquet file written to {output_file_path} for operator {operator}"
            )

    def merge_parquet_files_by_date(
        self,
        folder_a,
        folder_b,
        output_folder,
        timestamp_column="timestamp",
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
