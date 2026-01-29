import io
import os
from pathlib import Path
import shutil
import tarfile
from datetime import datetime, timezone
import json

import tqdm
from utils.data_pipeline_logger import DataPipelineLogger


class StationInformationTransformer:
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
            f"Starting StationInformation transformation for operator {self.operator}."
        )
        ndate = len(os.listdir(self.input_data_dir_path))
        for filename in tqdm.tqdm(
            os.listdir(self.input_data_dir_path), desc="Processing dates", total=ndate
        ):
            if filename.endswith(".tar.gz") or filename.endswith(".tgz"):
                input_path = os.path.join(self.input_data_dir_path, filename)
                output_path = os.path.join(self.export_data_dir_path, filename)

                with tarfile.open(input_path, "r:gz") as tar_in:
                    if len(tar_in.getmembers()) != 1:
                        self.logger.warning(
                            f"Too many files in '{filename}': expected 1 file, found {len(tar_in.getmembers())} for operator {self.operator}. Will only process the first file."
                        )

                    output_written = False
                    all_files_copied = False
                    # change last_updated to rfc3339 and version to 3.0
                    member = tar_in.getmembers()[0]
                    if not member.name.endswith(".json"):
                        self.logger.warning(
                            f"Skipping '{filename}': expected a .json file, found '{member.name}' for operator {self.operator}."
                        )
                        continue
                    json_content = tar_in.extractfile(member).read()
                    # check that json is not empty
                    if not json_content.strip() or b'"httpCode"' in json_content:
                        self.logger.warning(
                            f"Skipping '{filename}': JSON file is empty or contains httpCode for operator {self.operator}."
                        )
                        continue
                    try:
                        data = json.loads(json_content)
                    except json.JSONDecodeError as e:
                        self.logger.error(
                            f"Skipping '{filename}': JSON decode error: {e} for operator {self.operator}."
                        )
                        os.remove(output_path)
                        continue

                    # if version is already 3.0, copy file directly
                    if "version" in data and data["version"] == "3.0":
                        self.logger.info(
                            f"'{filename}' is already version 3.0, copying file directly for operator {self.operator}."
                        )
                        shutil.copyfile(input_path, output_path)
                        all_files_copied = True
                        continue

                    with tarfile.open(output_path, "w:gz") as tar_out:
                        # update last_updated to rfc3339
                        if "last_updated" in data:
                            dt = datetime.fromtimestamp(
                                data["last_updated"], tz=timezone.utc
                            )
                            # Convert to 2023-07-17T13:34:13+02:00 format
                            data["last_updated"] = dt.strftime("%Y-%m-%dT%H:%M:%S%z")
                            # insert colon in timezone offset
                            data["last_updated"] = (
                                data["last_updated"][:-2]
                                + ":"
                                + data["last_updated"][-2:]
                            )

                        # update version to 3.0
                        data["version"] = "3.0"

                        # update stations[].name
                        if "stations" in data["data"]:
                            for station in data["data"]["stations"]:
                                if "name" in station:
                                    name_text = station["name"]
                                    station["name"] = [
                                        {"text": name_text, "language": "en"}
                                    ]
                                # update stations[].short_name if exists
                                if "short_name" in station:
                                    short_name_text = station["short_name"]
                                    station["short_name"] = [
                                        {"text": short_name_text, "language": "en"}
                                    ]
                                # drop stations[].vehicle_type_capacity
                                if "vehicle_type_capacity" in station:
                                    del station["vehicle_type_capacity"]
                                # drop stations[].vehicle_capacity
                                if "vehicle_capacity" in station:
                                    del station["vehicle_capacity"]

                        updated_json_content = json.dumps(data)
                        tar_info = tarfile.TarInfo(name=member.name)
                        tar_info.size = len(updated_json_content)
                        tar_out.addfile(
                            tar_info, io.BytesIO(bytes(updated_json_content, "utf-8"))
                        )
                        output_written = True

                if not output_written and not all_files_copied:
                    self.logger.warning(
                        f"Output file '{output_path}' is empty. Removing it for operator {self.operator}."
                    )
                    os.remove(output_path)
