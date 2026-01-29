from pathlib import Path
from bs4 import BeautifulSoup
import requests
import pyarrow as pa
import pyarrow.parquet as pq
import datetime
import os

from utils import DataPipelineLogger


class Holidays:
    def __init__(
        self,
        extension_data_dir_path,
        meta_data_dir_path,
        logs_data_dir_path,
        input_data_dir_path,
        years,
    ):
        self.extension_data_dir_path = Path(extension_data_dir_path) / "holidays"
        self.meta_data_dir_path = Path(meta_data_dir_path)
        self.log_file = Path(logs_data_dir_path) / "logs.log"
        self.years = years if years else []

        # Setup logger
        self.logger = DataPipelineLogger.get_logger(
            name=self.__class__.__name__, log_file_path=self.log_file
        )

    def run(self):
        """Download the page https://www.schulferien.org/deutschland/feiertage/<year>.html
        for all years in self.years and extract the holidays."""

        holidays = []
        for year in self.years:
            url = f"https://www.schulferien.org/deutschland/feiertage/{year}"
            response = requests.get(url, verify=False)
            soup = BeautifulSoup(response.text, "html.parser")
            contentbox = soup.find("table", class_="feiertage")
            for tr in contentbox.find_all("tr"):
                if tr.has_attr("class") and "row_panel" in tr["class"]:
                    name = tr.find("a").text.replace("*", "").strip()
                    tds = tr.find_all("td")
                    date_str = tds[1].text.strip().split("\n")[0].strip()[3:]
                    # Convert to datetime.date
                    date_obj = datetime.datetime.strptime(date_str, "%d.%m.%Y").date()
                    start_date = date_obj
                    end_date = date_obj
                    place = tds[2].text.strip()
                    holiday_type = tr["class"][-1][:-4]
                    if holiday_type == "gesetzlich":
                        holiday_type = "public_holiday"
                    elif holiday_type == "nicht_gesetzlich":
                        holiday_type = "not_public_holiday"
                    elif holiday_type == "ereignis":
                        holiday_type = "event"
                    if "alle BL" in place:
                        holidays.append(
                            [
                                name,
                                start_date,
                                "Rheinland-Pfalz",
                                end_date,
                                holiday_type,
                            ]
                        )
                        holidays.append(
                            [
                                name,
                                start_date,
                                "Baden-Württemberg",
                                end_date,
                                holiday_type,
                            ]
                        )
                    if "BW" in place:
                        holidays.append(
                            [
                                name,
                                start_date,
                                "Baden-Württemberg",
                                end_date,
                                holiday_type,
                            ]
                        )
                    if "RP" in place:
                        holidays.append(
                            [
                                name,
                                start_date,
                                "Rheinland-Pfalz",
                                end_date,
                                holiday_type,
                            ]
                        )
        # Convert list of lists to list of dictionaries for PyArrow
        holidays_dict = []
        for holiday in holidays:
            holidays_dict.append(
                {
                    "name": holiday[0],
                    "start_date": holiday[1],
                    "federal_state_name": holiday[2],
                    "end_date": holiday[3],
                    "type": holiday[4],
                }
            )

        # Write to Parquet
        table = pa.Table.from_pylist(
            holidays_dict,
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

        pq.write_table(table, f"{self.extension_data_dir_path}/holidays.parquet", compression="BROTLI")
