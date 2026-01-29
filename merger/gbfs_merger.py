import os
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils.data_pipeline_logger import DataPipelineLogger
from transformers.free_bike_status_23_to_30 import FreeBikeStatusTransformer
from transformers.geofencing_zones_23_to_30 import GeofencingZonesTransformer
from transformers.station_information_23_to_30 import StationInformationTransformer
from transformers.system_pricing_plans_23_to_30 import SystemPricingPlansTransformer
from transformers.vehicle_types_23_to_30 import VehicleTypesTransformer
from merger.parquet_merger_gbfs import ParquetMergerGBFS


class GBFSMerger:
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
        input_1 = os.listdir(self.input_data_dir_path_1)
        input_2 = os.listdir(self.input_data_dir_path_2)

        uncommon_operators = set(input_1).symmetric_difference(set(input_2))
        common_operators = set(input_1).intersection(set(input_2))

        # Process common operators in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}

            for operator in common_operators:
                future = executor.submit(
                    self.merge_operator,
                    Path(self.input_data_dir_path_1, operator),
                    Path(self.input_data_dir_path_2, operator),
                    Path(self.export_data_dir_path, operator),
                    operator,
                )
                futures[future] = operator

            for operator in uncommon_operators:
                if operator in input_1:
                    future = executor.submit(
                        self.merge_operator,
                        Path(self.input_data_dir_path_1, operator),
                        Path(self.input_data_dir_path_1, operator),
                        Path(self.export_data_dir_path, operator),
                        operator,
                    )
                else:
                    future = executor.submit(
                        self.merge_operator,
                        Path(self.input_data_dir_path_2, operator),
                        Path(self.input_data_dir_path_2, operator),
                        Path(self.export_data_dir_path, operator),
                        operator,
                    )
                futures[future] = operator

            # Wait for all tasks to complete
            for future in as_completed(futures):
                operator = futures[future]
                future.result()
                self.logger.info(f"Successfully processed operator: {operator}")

    def merge_operator(self, input_dir_1, input_dir_2, output_dir, operator_name=None):
        if operator_name:
            self.logger.info(f"Starting merge for operator: {operator_name}")
        inputer_1 = str(hash(input_dir_1))
        inputer_2 = str(hash(input_dir_2))

        os.makedirs(os.path.join(self.tmp_dir, inputer_1), exist_ok=True)
        os.makedirs(os.path.join(self.tmp_dir, inputer_2), exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        self.update_to_v3(input_dir_1, inputer_1, operator_name=operator_name)
        self.update_to_v3(input_dir_2, inputer_2, operator_name=operator_name)

        self.merge_targz(
            os.path.join(self.tmp_dir, inputer_1),
            os.path.join(self.tmp_dir, inputer_2),
            os.path.join(output_dir),
            operator_name=operator_name,
        )

        # check if temp dir free_bike_status_v30 has data
        free_bike_status_dir_1 = os.path.join(
            self.tmp_dir, inputer_1, "free_bike_status_v30"
        )
        free_bike_status_dir_2 = os.path.join(
            self.tmp_dir, inputer_2, "free_bike_status_v30"
        )
        if (
            os.path.exists(free_bike_status_dir_1)
            and os.path.exists(Path(input_dir_1, "vehicle_status"))
            and len(os.listdir(free_bike_status_dir_1)) > 0
            and len(os.listdir(Path(input_dir_1, "vehicle_status"))) > 0
        ):
            self.logger.info(
                f"Merging vehicle_status from {input_dir_1} and {input_dir_2} to {output_dir} for operator {operator_name}"
            )
            ParquetMergerGBFS(
                log_file_path=self.log_file
            ).merge_parquet_files_by_date(
                free_bike_status_dir_1,
                Path(input_dir_1, "vehicle_status"),
                os.path.join(self.tmp_dir, inputer_1, "vehicle_status"),
                operator=operator_name,
            )
        elif (
            os.path.exists(free_bike_status_dir_1)
            and len(os.listdir(free_bike_status_dir_1)) > 0
        ):
            self.logger.info(
                f"Copying vehicle_status from {free_bike_status_dir_1} to {os.path.join(self.tmp_dir, inputer_1, 'vehicle_status')} for operator {operator_name}"
            )
            shutil.copytree(
                free_bike_status_dir_1,
                os.path.join(self.tmp_dir, inputer_1, "vehicle_status"),
                dirs_exist_ok=True,
            )
        elif (
            os.path.exists(Path(input_dir_1, "vehicle_status"))
            and len(os.listdir(Path(input_dir_1, "vehicle_status"))) > 0
        ):
            self.logger.info(
                f"Copying vehicle_status from {Path(input_dir_1, 'vehicle_status')} to {os.path.join(self.tmp_dir, inputer_1, 'vehicle_status')} for operator {operator_name}"
            )
            shutil.copytree(
                Path(input_dir_1, "vehicle_status"),
                os.path.join(self.tmp_dir, inputer_1, "vehicle_status"),
                dirs_exist_ok=True,
            )
        else:
            self.logger.error(
                f"No vehicle_status data found for operator {input_dir_1} for operator {operator_name}"
            )
            return

        if (
            os.path.exists(free_bike_status_dir_2)
            and os.path.exists(Path(input_dir_2, "vehicle_status"))
            and len(os.listdir(free_bike_status_dir_2)) > 0
            and len(os.listdir(Path(input_dir_2, "vehicle_status"))) > 0
        ):
            self.logger.info(
                f"Merging vehicle_status from {input_dir_1} and {input_dir_2} to {output_dir} for operator {operator_name}"
            )
            ParquetMergerGBFS(
                log_file_path=self.log_file
            ).merge_parquet_files_by_date(
                free_bike_status_dir_2,
                Path(input_dir_2, "vehicle_status"),
                os.path.join(self.tmp_dir, inputer_2, "vehicle_status"),
                operator=operator_name,
            )
        elif (
            os.path.exists(free_bike_status_dir_2)
            and len(os.listdir(free_bike_status_dir_2)) > 0
        ):
            self.logger.info(
                f"Copying vehicle_status from {free_bike_status_dir_2} to {os.path.join(self.tmp_dir, inputer_2, 'vehicle_status')} for operator {operator_name}"
            )
            shutil.copytree(
                free_bike_status_dir_2,
                os.path.join(self.tmp_dir, inputer_2, "vehicle_status"),
                dirs_exist_ok=True,
            )
        elif (
            os.path.exists(Path(input_dir_2, "vehicle_status"))
            and len(os.listdir(Path(input_dir_2, "vehicle_status"))) > 0
        ):
            self.logger.info(
                f"Copying vehicle_status from {Path(input_dir_2, 'vehicle_status')} to {os.path.join(self.tmp_dir, inputer_2, 'vehicle_status')} for operator {operator_name}"
            )
            shutil.copytree(
                Path(input_dir_2, "vehicle_status"),
                os.path.join(self.tmp_dir, inputer_2, "vehicle_status"),
                dirs_exist_ok=True,
            )
        else:
            self.logger.error(
                f"No vehicle_status data found for operator {input_dir_2} for operator {operator_name}"
            )
            return

        ParquetMergerGBFS(log_file_path=self.log_file).merge_parquet_files_by_date(
            os.path.join(self.tmp_dir, inputer_1, "vehicle_status"),
            os.path.join(self.tmp_dir, inputer_2, "vehicle_status"),
            os.path.join(output_dir, "vehicle_status"),
            operator=operator_name,
        )

        # clear tmp dir and recreate
        if os.path.exists(os.path.join(self.tmp_dir, inputer_1)):
            shutil.rmtree(os.path.join(self.tmp_dir, inputer_1))
        if os.path.exists(os.path.join(self.tmp_dir, inputer_2)):
            shutil.rmtree(os.path.join(self.tmp_dir, inputer_2))

    def update_to_v3(self, operator_dir, inputer, operator_name=None):
        if os.path.exists(os.path.join(operator_dir, "geofencing_zones/")):
            self.logger.info(
                f"Updating geofencing_zones for operator {operator_dir} to v3.0 for operator {operator_name}"
            )
            GeofencingZonesTransformer(
                os.path.join(operator_dir, "geofencing_zones/"),
                os.path.join(self.tmp_dir, inputer, "geofencing_zones_v30/"),
                self.logs_data_dir_path,
                operator=operator_name,
            ).run_transformer()

        if os.path.exists(os.path.join(operator_dir, "station_information/")):
            self.logger.info(
                f"Updating station_information for operator {operator_dir} to v3.0 for operator {operator_name}"
            )
            StationInformationTransformer(
                os.path.join(operator_dir, "station_information/"),
                os.path.join(self.tmp_dir, inputer, "station_information_v30/"),
                self.logs_data_dir_path,
                operator=operator_name,
            ).run_transformer()
        if os.path.exists(os.path.join(operator_dir, "system_pricing_plans/")):
            self.logger.info(
                f"Updating system_pricing_plans for operator {operator_dir} to v3.0 for operator {operator_name}"
            )
            SystemPricingPlansTransformer(
                os.path.join(operator_dir, "system_pricing_plans/"),
                os.path.join(self.tmp_dir, inputer, "system_pricing_plans_v30/"),
                self.logs_data_dir_path,
                operator=operator_name,
            ).run_transformer()
        if os.path.exists(os.path.join(operator_dir, "vehicle_types/")):
            self.logger.info(
                f"Updating vehicle_types for operator {operator_dir} to v3.0 for operator {operator_name}"
            )
            VehicleTypesTransformer(
                os.path.join(operator_dir, "vehicle_types/"),
                os.path.join(self.tmp_dir, inputer, "vehicle_types_v30/"),
                self.logs_data_dir_path,
                operator=operator_name,
            ).run_transformer()
        if os.path.exists(os.path.join(operator_dir, "free_bike_status/")):
            self.logger.info(
                f"Updating free_bike_status for operator {operator_dir} to v3.0 for operator {operator_name}"
            )
            FreeBikeStatusTransformer(
                os.path.join(operator_dir, "free_bike_status/"),
                os.path.join(self.tmp_dir, inputer, "free_bike_status_v30/"),
                self.logs_data_dir_path,
                operator=operator_name,
            ).run_transformer()
        self.logger.info(
            f"Finished updating operator {operator_dir} to v3.0 for operator {operator_name}"
        )

    def merge_targz(self, input_dir_1, input_dir_2, output_dir, operator_name=None):

        targets = [
            "geofencing_zones_v30",
            "station_information_v30",
            "system_pricing_plans_v30",
            "vehicle_types_v30",
        ]
        for target in targets:
            if os.path.exists(os.path.join(input_dir_1, target)) and os.path.exists(
                os.path.join(input_dir_2, target)
            ):
                self.logger.info(
                    f"Merging {target} from {input_dir_1} and {input_dir_2} to {output_dir} for operator {operator_name}"
                )
                dir_1 = os.path.join(input_dir_1, target)
                dir_2 = os.path.join(input_dir_2, target)
                output_target_dir = os.path.join(output_dir, target.replace("_v30", ""))
                os.makedirs(output_target_dir, exist_ok=True)
                # copy files from input_dir_1 to output_dir
                shutil.copytree(dir_1, output_target_dir, dirs_exist_ok=True)
                # files not in input_dir_1
                files_not_in_1 = set(os.listdir(dir_2)) - set(os.listdir(dir_1))
                # copy those from input_dir_2 to output_dir
                for filename in files_not_in_1:
                    shutil.copy2(
                        os.path.join(dir_2, filename),
                        os.path.join(output_target_dir, filename),
                    )
            elif os.path.exists(os.path.join(input_dir_1, target)):
                self.logger.info(f"Copying {target} from {input_dir_1} to {output_dir}")
                dir_1 = os.path.join(input_dir_1, target)
                output_target_dir = os.path.join(output_dir, target.replace("_v30", ""))
                shutil.copytree(dir_1, output_target_dir, dirs_exist_ok=True)
            elif os.path.exists(os.path.join(input_dir_2, target)):
                self.logger.info(f"Copying {target} from {input_dir_2} to {output_dir}")
                dir_2 = os.path.join(input_dir_2, target)
                output_target_dir = os.path.join(output_dir, target.replace("_v30", ""))
                shutil.copytree(dir_2, output_target_dir, dirs_exist_ok=True)
            else:
                self.logger.warning(
                    f"No data found for {target} in either {input_dir_1} or {input_dir_2} for operator {operator_name}"
                )
        self.logger.info(
            f"Finished merging targz from {input_dir_1} and {input_dir_2} to {output_dir} for operator {operator_name}"
        )
