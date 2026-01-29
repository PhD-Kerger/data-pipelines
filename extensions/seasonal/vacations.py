from pathlib import Path
from bs4 import BeautifulSoup
import requests
import pyarrow as pa
import pyarrow.parquet as pq
import datetime
import os

from utils import DataPipelineLogger


class Vacations:
    def __init__(
        self,
        extension_data_dir_path,
        meta_data_dir_path,
        logs_data_dir_path,
        input_data_dir_path,
        years,
    ):
        self.extension_data_dir_path = Path(extension_data_dir_path) / "vacations"
        self.meta_data_dir_path = Path(meta_data_dir_path)
        self.log_file = Path(logs_data_dir_path) / "logs.log"

        # Setup logger
        self.logger = DataPipelineLogger.get_logger(
            name=self.__class__.__name__, log_file_path=self.log_file
        )

        self.years = years if years else []

    def run(self):
        """Download the page https://www.schulferien.org/deutschland/ferien/<year>.html
        for all years in self.years and extract the vacations."""
        vacations = []
        for year in self.years:
            url = f"https://www.schulferien.org/deutschland/ferien/{year}"
            response = requests.get(url, verify=False)
            soup = BeautifulSoup(response.text, "html.parser")
            contentbox = soup.find("div", class_="contentbox")
            for tr in contentbox.find_all("tr"):
                tds = tr.find_all("td")
                for td in tds[1:]:
                    name = td.get("data-header")
                    federal_state = tds[0].text.replace("*", "").strip()
                    if federal_state in ["Baden-Württemberg", "Rheinland-Pfalz"]:
                        date = td.text.replace("*", "").strip()
                        if "+" in date:
                            date = date.split("+")[1]
                        if date == "-":
                            continue
                        elif "-" not in date:
                            start_date_str = date + f"{year}"
                            end_date_str = date + f"{year}"
                        else:
                            start_date_str = date.split("-")[0].strip() + f"{year}"
                            end_date_str = date.split("-")[1].strip() + f"{year}"
                        # Convert to datetime.date
                        start_date = datetime.datetime.strptime(
                            start_date_str, "%d.%m.%Y"
                        ).date()
                        end_date = datetime.datetime.strptime(
                            end_date_str, "%d.%m.%Y"
                        ).date()
                        vacations.append(
                            [
                                name,
                                start_date,
                                federal_state,
                                end_date,
                                "school_vacation",
                            ]
                        )
        # Convert list of lists to list of dictionaries for PyArrow
        vacations_dict = []
        for vacation in vacations:
            vacations_dict.append(
                {
                    "name": vacation[0],
                    "start_date": vacation[1],
                    "federal_state_name": vacation[2],
                    "end_date": vacation[3],
                    "type": vacation[4],
                }
            )

        # Write to Parquet
        table = pa.Table.from_pylist(
            vacations_dict,
            schema=pa.schema(
                [
                    ("name", pa.string()),
                    ("start_date", pa.date32()),
                    ("federal_state_name", pa.string()),
                    ("end_date", pa.date32()),
                    ("type", pa.string()),
                ]
            ),
        )

        os.makedirs(self.extension_data_dir_path, exist_ok=True)

        pq.write_table(
            table, f"{self.extension_data_dir_path}/school_vacations.parquet", compression="BROTLI"
        )
