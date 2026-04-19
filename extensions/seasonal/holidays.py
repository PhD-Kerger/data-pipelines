import os
from pathlib import Path
import requests
import pyarrow as pa
import pyarrow.parquet as pq
import datetime


from utils import DataPipelineLogger


class Holidays:
    def __init__(
        self,
        extension_data_dir_path,
        meta_data_dir_path,
        logs_data_dir_path,
        input_data_dir_path,
        from_date,
        to_date,
        country_iso_codes,
    ):
        self.extension_data_dir_path = Path(extension_data_dir_path) / "holidays"
        self.meta_data_dir_path = Path(meta_data_dir_path)
        self.log_file = Path(logs_data_dir_path) / "logs.log"
        self.from_date = from_date
        self.to_date = to_date
        self.country_iso_codes = country_iso_codes

        # Setup logger
        self.logger = DataPipelineLogger.get_logger(
            name=self.__class__.__name__, log_file_path=self.log_file
        )

    def run(self):
        self.logger.info("Starting Holidays data pipeline.")
        public_holidays = []
        school_holidays = []
        for country in self.country_iso_codes:
            url = f"https://openholidaysapi.org/PublicHolidays?countryIsoCode={country}&languageIsoCode=EN&validFrom={self.from_date}&validTo={self.to_date}"
            self.logger.info(f"Fetching holidays for country: {country}")
            try:
                response = requests.get(url)
                response.raise_for_status()
                country_holidays = response.json()
                for holiday in country_holidays:
                    if holiday.get("subdivisions"):
                        for subdivision in holiday.get("subdivisions"):
                            public_holidays.append(
                                {
                                    "name": holiday.get("name")[0].get("text"),
                                    "start_date": datetime.datetime.strptime(
                                        holiday.get("startDate"), "%Y-%m-%d"
                                    ).date(),
                                    "country_name": country,
                                    "federal_state_name": self.subdivision_code_mapper(
                                        subdivision.get("code")
                                    ),
                                    "end_date": datetime.datetime.strptime(
                                        holiday.get("endDate"), "%Y-%m-%d"
                                    ).date(),
                                    "type": holiday.get("type"),
                                }
                            )
                    else:
                        public_holidays.append(
                            {
                                "name": holiday.get("name")[0].get("text"),
                                "start_date": datetime.datetime.strptime(
                                    holiday.get("startDate"), "%Y-%m-%d"
                                ).date(),
                                "country_name": country,
                                "federal_state_name": None,
                                "end_date": datetime.datetime.strptime(
                                    holiday.get("endDate"), "%Y-%m-%d"
                                ).date(),
                                "type": holiday.get("type"),
                            }
                        )
            except requests.exceptions.RequestException as e:
                self.logger.error(
                    f"Error fetching holidays for country: {country}. Error: {e}"
                )
            url = f"https://openholidaysapi.org/SchoolHolidays?countryIsoCode={country}&languageIsoCode=EN&validFrom={self.from_date}&validTo={self.to_date}"
            self.logger.info(f"Fetching school holidays for country: {country}")
            try:
                response = requests.get(url)
                response.raise_for_status()
                country_school_holidays = response.json()
                for holiday in country_school_holidays:
                    if holiday.get("subdivisions"):
                        for subdivision in holiday.get("subdivisions"):
                            school_holidays.append(
                                {
                                    "name": holiday.get("name")[0].get("text"),
                                    "start_date": datetime.datetime.strptime(
                                        holiday.get("startDate"), "%Y-%m-%d"
                                    ).date(),
                                    "country_name": country,
                                    "federal_state_name": self.subdivision_code_mapper(
                                        subdivision.get("code")
                                    ),
                                    "end_date": datetime.datetime.strptime(
                                        holiday.get("endDate"), "%Y-%m-%d"
                                    ).date(),
                                    "type": holiday.get("type"),
                                }
                            )
                    else:
                        school_holidays.append(
                            {
                                "name": holiday.get("name")[0].get("text"),
                                "start_date": datetime.datetime.strptime(
                                    holiday.get("startDate"), "%Y-%m-%d"
                                ).date(),
                                "country_name": country,
                                "federal_state_name": None,
                                "end_date": datetime.datetime.strptime(
                                    holiday.get("endDate"), "%Y-%m-%d"
                                ).date(),
                                "type": holiday.get("type"),
                            }
                        )
            except requests.exceptions.RequestException as e:
                self.logger.error(
                    f"Error fetching school holidays for country: {country}. Error: {e}"
                )

        holidays_dict = public_holidays + school_holidays

        # Write to Parquet
        table = pa.Table.from_pylist(
            holidays_dict,
            schema=pa.schema(
                [
                    ("name", pa.string()),
                    ("start_date", pa.date32()),
                    ("country_name", pa.string()),
                    ("federal_state_name", pa.string()),
                    ("end_date", pa.date32()),
                    ("type", pa.string()),
                ]
            ),
        )

        os.makedirs(self.extension_data_dir_path, exist_ok=True)

        pq.write_table(
            table,
            f"{self.extension_data_dir_path}/holidays.parquet",
            compression="BROTLI",
        )

    def subdivision_code_mapper(self, subdivision_code):
        """Maps subdivision codes to their correct corresponding names.
        Sometimes the API returns incorrect subdivision codes, so this function
        is used to map them to the correct names that are used in the rest of the pipeline.
        """
        germany_mapping = {
            "DE-BY-AU": "DE-BY",
        }

        austria_mapping = {
            "AT-BL": "AT-1",
            "AT-KÄ": "AT-2",
            "AT-NÖ": "AT-3",
            "AT-OÖ": "AT-4",
            "AT-SB": "AT-5",
            "AT-SM": "AT-6",
            "AT-TI": "AT-7",
            "AT-VA": "AT-8",
            "AT-WI": "AT-9",
        }
        spain_mapping = {
            "ES-CN-LP-LA": "ES-CN",
            "ES-CN-SC-TE": "ES-CN",
            "ES-CN-SC-LG": "ES-CN",
            "ES-CN-SC-EH": "ES-CN",
            "ES-CN-LP-GC": "ES-CN",
            "ES-CN-LP-FV": "ES-CN",
            "ES-AN-SE": "ES-AN",
            "ES-AN-CO": "ES-AN",
            "ES-AN-JA": "ES-AN",
            "ES-AN-HL": "ES-AN",
            "ES-AN-MA": "ES-AN",
            "ES-AN-GR": "ES-AN",
            "ES-AR-HU": "ES-AR",
            "ES-AR-ZG": "ES-AR",
            "ES-AR-TE": "ES-AR",
        }
        poland_mapping = {
            "PL-DS": "PL-02",
            "PL-KP": "PL-04",
            "PL-LU": "PL-06",
            "PL-LB": "PL-08",
            "PL-LD": "PL-10",
            "PL-MA": "PL-12",
            "PL-MZ": "PL-14",
            "PL-OP": "PL-16",
            "PL-PK": "PL-18",
            "PL-PD": "PL-20",
            "PL-PM": "PL-22",
            "PL-SL": "PL-24",
            "PL-SK": "PL-26",
            "PL-WN": "PL-28",
            "PL-WP": "PL-30",
            "PL-ZP": "PL-32",
        }
        czech_region_mapping = {
            "CZ-PR": "CZ-10",
            "CZ-ST": "CZ-20",
            "CZ-JC": "CZ-31",
            "CZ-PL": "CZ-32",
            "CZ-KA": "CZ-41",
            "CZ-US": "CZ-42",
            "CZ-LI": "CZ-51",
            "CZ-KR": "CZ-52",
            "CZ-PA": "CZ-53",
            "CZ-VY": "CZ-63",
            "CZ-JM": "CZ-64",
            "CZ-OL": "CZ-71",
            "CZ-ZL": "CZ-72",
            "CZ-MO": "CZ-80",
        }
        france_mapping = {
            # Legacy/alias metro region codes -> current ISO region codes
            "FR-AR": "FR-ARA",
            "FR-BF": "FR-BFC",
            "FR-BT": "FR-BRE",
            "FR-CV": "FR-CVL",
            "FR-GE": "FR-GES",
            "FR-GE-BR": "FR-GES",
            "FR-GE-HR": "FR-GES",
            "FR-GE-MO": "FR-GES",
            "FR-HF": "FR-HDF",
            "FR-IF": "FR-IDF",
            "FR-NO": "FR-NOR",
            "FR-NA": "FR-NAQ",
            "FR-OC": "FR-OCC",
            "FR-PL": "FR-PDL",
            "FR-SP": "FR-PAC",
            "FR-BL": None,
            "FR-CO": None,
            "FR-GP": None,
            "FR-GY": None,
            "FR-MF": None,
            "FR-MQ": None,
            "FR-PC": None,
            "FR-RU": None,
            "FR-YT": None,
            "FR-ZA": None,
            "FR-ZB": None,
            "FR-ZC": None,
        }
        slovenia_mapping = {
            # Cohesion/statistical macro-regions, not ISO municipality subdivision codes.
            "SI-VR": None,
            "SI-ZR": None,
        }
        italy_region_mapping = {
            "IT-AB": "IT-65",
            "IT-VA": "IT-23",
            "IT-PU": "IT-75",
            "IT-BA": "IT-77",
            "IT-ER": "IT-45",
            "IT-FV": "IT-36",
            "IT-CL": "IT-78",
            "IT-CM": "IT-72",
            "IT-LA": "IT-62",
            "IT-LI": "IT-42",
            "IT-LO": "IT-25",
            "IT-MA": "IT-57",
            "IT-MO": "IT-67",
            "IT-PI": "IT-21",
            "IT-SA": "IT-88",
            "IT-SI": "IT-82",
            "IT-TO": "IT-52",
            "IT-TR": "IT-32",
            "IT-UM": "IT-55",
            "IT-VE": "IT-34",
        }
        portugal_region_mapping = {
            "PT-AV": "PT-01",
            "PT-BE": "PT-02",
            "PT-BR": "PT-03",
            "PT-BA": "PT-04",
            "PT-CB": "PT-05",
            "PT-CO": "PT-06",
            "PT-EV": "PT-07",
            "PT-FA": "PT-08",
            "PT-GU": "PT-09",
            "PT-LE": "PT-10",
            "PT-LI": "PT-11",
            "PT-PA": "PT-12",
            "PT-PO": "PT-13",
            "PT-SA": "PT-14",
            "PT-SE": "PT-15",
            "PT-VC": "PT-16",
            "PT-VR": "PT-17",
            "PT-VI": "PT-18",
            "PT-AC": "PT-20",
            "PT-MA": "PT-30",
        }
        swiss_canton_codes = {
            "CH-AG",
            "CH-AR",
            "CH-AI",
            "CH-BL",
            "CH-BS",
            "CH-BE",
            "CH-FR",
            "CH-GE",
            "CH-GL",
            "CH-GR",
            "CH-JU",
            "CH-LU",
            "CH-NE",
            "CH-NW",
            "CH-OW",
            "CH-SH",
            "CH-SZ",
            "CH-SO",
            "CH-SG",
            "CH-TI",
            "CH-TG",
            "CH-UR",
            "CH-VD",
            "CH-VS",
            "CH-ZG",
            "CH-ZH",
        }
        mapping = {
            **austria_mapping,
            **spain_mapping,
            **germany_mapping,
            **poland_mapping,
            **france_mapping,
            **slovenia_mapping,
            **italy_region_mapping,
            **portugal_region_mapping,
        }

        if subdivision_code in mapping:
            return mapping[subdivision_code]

        # Czech API responses can include district-level codes (e.g. CZ-US-UL).
        # Normalize them to the parent region code (e.g. CZ-42).
        if subdivision_code and subdivision_code.startswith("CZ-"):
            czech_prefix = "-".join(subdivision_code.split("-")[:2])
            if czech_prefix in czech_region_mapping:
                return czech_region_mapping[czech_prefix]

        # Italian API responses can include province-level suffixes
        # (e.g. IT-ER-BO, IT-TO-FI). Normalize to region code.
        if subdivision_code and subdivision_code.startswith("IT-"):
            italy_prefix = "-".join(subdivision_code.split("-")[:2])
            if italy_prefix in italy_region_mapping:
                return italy_region_mapping[italy_prefix]

        # Portuguese API responses can include municipality-level suffixes
        # (e.g. PT-AV-CP, PT-LI-OE). Normalize to district/region code.
        if subdivision_code and subdivision_code.startswith("PT-"):
            portugal_prefix = "-".join(subdivision_code.split("-")[:2])
            if portugal_prefix in portugal_region_mapping:
                return portugal_region_mapping[portugal_prefix]

        # Swiss API responses can include district/municipality-level codes
        # (e.g. CH-AG-BA-BA). Normalize them to the canton code (e.g. CH-AG).
        if subdivision_code and subdivision_code.startswith("CH-"):
            swiss_prefix = "-".join(subdivision_code.split("-")[:2])
            if swiss_prefix in swiss_canton_codes:
                return swiss_prefix

        return subdivision_code
