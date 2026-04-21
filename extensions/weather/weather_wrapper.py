import os
from pathlib import Path

from extensions.weather.mannheim_weather_stations import MannheimWeatherStations
from extensions.weather.stuttgart_weather_stations import StuttgartWeatherStations
from utils.data_pipeline_logger import DataPipelineLogger
from extensions.weather.dwd import DWD
from extensions.weather.openmeteo import OpenMeteo


class Weather:
    """Wrapper class for different weather data sources."""

    # Mapping of weather source names to their corresponding classes
    WEATHER_SOURCE_MAPPING = {
        "dwd": DWD,
        "mannheim": MannheimWeatherStations,
        "stuttgart": StuttgartWeatherStations,
        "openmeteo": OpenMeteo,
    }

    def __init__(
        self,
        extension_data_dir_path,
        meta_data_dir_path,
        input_data_dir_path,
        logs_data_dir_path,
        start_date,
        end_date,
        locations,
        weather_source,
    ):
        # Setup logger
        self.logger = DataPipelineLogger.get_logger(
            name=self.__class__.__name__,
            log_file_path=Path(logs_data_dir_path) / "logs.log",
        )

        # Validate weather source
        if weather_source not in self.WEATHER_SOURCE_MAPPING:
            available_sources = ", ".join(self.WEATHER_SOURCE_MAPPING.keys())
            error_msg = f"Unknown weather source: {weather_source}. Available sources: {available_sources}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Get the appropriate weather class
        WeatherClass = self.WEATHER_SOURCE_MAPPING[weather_source]

        self.logger.info(f"Initializing weather source: {weather_source}")

        # Instantiate the weather data source
        self.weather_instance = WeatherClass(
            extension_data_dir_path=extension_data_dir_path,
            meta_data_dir_path=meta_data_dir_path,
            input_data_dir_path=input_data_dir_path,
            logs_data_dir_path=logs_data_dir_path,
            start_date=start_date,
            end_date=end_date,
            locations=locations,
        )

    def run(self):
        """Execute the weather data processing for the selected source."""
        self.logger.info("Starting weather data processing")
        self.weather_instance.run()
        self.logger.info("Weather data processing completed")
