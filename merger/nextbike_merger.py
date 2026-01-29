import os
import shutil
from pathlib import Path
from merger.parquet_merger_nextbike import ParquetMergerNextbike
from utils.data_pipeline_logger import DataPipelineLogger


class NextbikeMerger:
    def __init__(
        self,
        input_data_dir_path_1,
        input_data_dir_path_2,
        export_data_dir_path,
        logs_data_dir_path,
        tmp_dir_path,
    ):
        if not hasattr(self, "initialized"):
            self.initialized = True
            self.input_data_dir_path_1 = input_data_dir_path_1
            self.input_data_dir_path_2 = input_data_dir_path_2
            self.tmp_dir = tmp_dir_path
            self.export_data_dir_path = export_data_dir_path
            self.logs_data_dir_path = logs_data_dir_path
            self.log_file = Path(logs_data_dir_path) / "logs.log"
            self.max_workers = os.cpu_count() or 4
            # Setup logger
            self.logger = DataPipelineLogger.get_logger(
                name=self.__class__.__name__, log_file_path=self.log_file
            )

    def run(self):
        self.logger.info(f"Processing operator nextbike")
        if os.path.exists(self.export_data_dir_path):
            shutil.rmtree(self.export_data_dir_path)
        ParquetMergerNextbike(log_file_path=self.log_file).merge_parquet_files_by_date(
            folder_a=Path(self.input_data_dir_path_1),
            folder_b=Path(self.input_data_dir_path_2),
            output_folder=Path(self.export_data_dir_path),
            operator="Nextbike",
        )
        self.logger.info(f"Successfully processed operator nextbike")
