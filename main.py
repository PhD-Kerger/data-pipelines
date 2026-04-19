from pathlib import Path
import yaml

from operators import Nextbike, Tier, GBFS
from extensions import (
    Holidays,
    OSM,
    OSMLanduse,
    Geo,
    GTFS,
    Foursquare,
    Demographics_MA,
    WFS,
    BikeCountStationsGermany,
    Weather,
)
from transformers import (
    FreeBikeStatusTransformer,
    SystemPricingPlansTransformer,
    VehicleTypesTransformer,
    GeofencingZonesTransformer,
    StationInformationTransformer,
)

from merger import GBFSMerger, NextbikeMerger


from utils import DataPipelineLogger


class DataPipelineManager:
    """Main class for managing data pipeline operations."""

    def __init__(self, config_path="env.yaml"):
        """Initialize the DataPipelineManager with configuration and logger."""
        self.config_path = config_path
        self.config = self.load_config()

        # Extract configuration sections
        processing_config = self.config.get("processing", {})
        self.directories = processing_config.get("directories", {})

        self.mergers_config = (
            processing_config.get("mergers", [])
            if processing_config.get("mergers")
            else []
        )
        self.transformers_config = (
            processing_config.get("transformers", [])
            if processing_config.get("transformers")
            else []
        )
        self.processors_config = (
            processing_config.get("processors", [])
            if processing_config.get("processors")
            else []
        )
        self.extension_config = (
            processing_config.get("extensions", [])
            if processing_config.get("extensions")
            else []
        )

        self.osrm_config = self.config.get("osrm", {})
        self.OSRM_ENABLED = self.osrm_config.get("enabled", False)
        self.OSRM_ALTERNATIVE_PERCENTAGE = self.osrm_config.get(
            "alternative_percentage", 0
        )

        # Setup logger as class attribute
        self.logger = DataPipelineLogger.get_logger(
            name=self.__class__.__name__,
            log_file_path=Path(self.directories.get("logs", "logs")) / "logs.log",
        )

        # Class mappings for processors and extensions and transformers
        self.processor_class_mapping = {
            "GBFS": GBFS,
            "Tier": Tier,
            "Nextbike": Nextbike,
        }

        self.extension_class_mapping = {
            "Holidays": Holidays,
            "OSM": OSM,
            "OSMLanduse": OSMLanduse,
            "Weather": Weather,
            "Geo": Geo,
            "GTFS": GTFS,
            "Foursquare": Foursquare,
            "Demographics-MA": Demographics_MA,
            "WFS": WFS,
            "BikeCountStationsGermany": BikeCountStationsGermany,
        }

        self.transformer_class_mapping = {
            "StationInformation": StationInformationTransformer,
            "FreeBikeStatus": FreeBikeStatusTransformer,
            "SystemPricingPlans": SystemPricingPlansTransformer,
            "VehicleTypes": VehicleTypesTransformer,
            "GeofencingZones": GeofencingZonesTransformer,
        }

        self.merger_class_mapping = {
            "GBFSMerger": GBFSMerger,
            "NextbikeMerger": NextbikeMerger,
        }

    def load_config(self):
        """Load configuration from YAML file."""
        with open(self.config_path, "r") as file:
            return yaml.safe_load(file)

    def create_merger(self, merger_name, merger_config):
        """Create a merger instance based on configuration."""
        merger_class = merger_config.get("class")

        if merger_class not in self.merger_class_mapping:
            raise ValueError(f"Unknown merger class: {merger_class}")

        # Get the merger class constructor
        MergerClass = self.merger_class_mapping[merger_class]

        return MergerClass(
            logs_data_dir_path=self.directories.get("logs", "logs"),
            **merger_config.get("config", {}),
        )

    def create_transformer(self, transformer_name, transformer_config):
        """Create a transformer instance based on configuration."""
        transformer_class = transformer_config.get("class")

        if transformer_class not in self.transformer_class_mapping:
            raise ValueError(f"Unknown transformer class: {transformer_class}")

        # Get the transformer class constructor
        TransformerClass = self.transformer_class_mapping[transformer_class]

        return TransformerClass(
            logs_data_dir_path=self.directories.get("logs", "logs"),
            **transformer_config.get("config", {}),
        )

    def create_processor(
        self,
        processor_name,
        processor_config,
        processing_steps,
        n_vehicles=None,
        n_entries=None,
    ):
        """Create a processor instance based on configuration."""
        processor_class = processor_config.get("class")

        if processor_class not in self.processor_class_mapping:
            raise ValueError(f"Unknown processor class: {processor_class}")

        # Get the processor class constructor
        ProcessorClass = self.processor_class_mapping[processor_class]

        # Create processor instance dynamically
        return ProcessorClass(
            input_data_dir_path=f"{self.directories.get('input', 'input')}/{processor_name}",
            meta_data_dir_path=self.directories.get("metadata", "metadata"),
            export_data_dir_path=f"{self.directories.get('output', 'output')}/{processor_name}",
            logs_data_dir_path=self.directories.get("logs", "logs"),
            osrm_enabled=self.OSRM_ENABLED,
            osrm_alternative_percentage=self.OSRM_ALTERNATIVE_PERCENTAGE,
            processing_steps=processing_steps,
            processor_class=processor_class,
            **processor_config.get("config", {}),
            n_vehicles=n_vehicles,
            n_entries=n_entries,
        )

    def create_extension(self, extension_name, extension_config):
        """Create an extension instance based on configuration."""
        extension_class = extension_config.get("class")

        if extension_class not in self.extension_class_mapping:
            raise ValueError(f"Unknown extension class: {extension_class}")

        # Get the extension class constructor
        ExtensionClass = self.extension_class_mapping[extension_class]

        # Create extension instance dynamically
        return ExtensionClass(
            extension_data_dir_path=self.directories.get("extensions", "extensions"),
            input_data_dir_path=f"{self.directories.get('input', 'input')}/{extension_name}",
            meta_data_dir_path=self.directories.get("metadata", "metadata"),
            logs_data_dir_path=self.directories.get("logs", "logs"),
            **extension_config.get("config", {}),
        )

    def process_transformers(self):
        """Run all configured transformers."""
        self.logger.info(
            f"Found {len(self.transformers_config)} transformers to process"
        )

        # Process each configured transformer
        for transformer_item in self.transformers_config:
            # Extract transformer name and configuration
            transformer_name = list(transformer_item.keys())[0]
            transformer_config = transformer_item[transformer_name]

            transformer_class = transformer_config.get("class", "unknown")
            self.logger.info(
                f"Creating {transformer_class} transformer: {transformer_name}"
            )

            try:
                # Create and run transformer instance
                transformer = self.create_transformer(
                    transformer_name, transformer_config
                )
                transformer.run()

                self.logger.info(f"Completed {transformer_name} transformer")

            except Exception as e:
                self.logger.error(f"Error processing {transformer_name}: {str(e)}")
                continue

        self.logger.info("All transformers completed!")

    def process_mergers(self):
        """Run all configured mergers."""
        self.logger.info(f"Found {len(self.mergers_config)} mergers to process")

        # Process each configured merger
        for merger_item in self.mergers_config:
            # Extract merger name and configuration
            merger_name = list(merger_item.keys())[0]
            merger_config = merger_item[merger_name]

            merger_class = merger_config.get("class", "unknown")
            self.logger.info(f"Creating {merger_class} merger: {merger_name}")

            try:
                # Create and run merger instance
                merger = self.create_merger(merger_name, merger_config)
                merger.run()

                self.logger.info(f"Completed {merger_name} merger")

            except Exception as e:
                self.logger.error(f"Error processing {merger_name}: {str(e)}")
                continue

        self.logger.info("All mergers completed!")

    def run_generators(
        self,
        processor: Tier | GBFS | Nextbike,
        processor_name: str,
    ):
        """Run all generators for a processor."""
        # Generate trips data
        if "trips" in processor.getProcessingSteps():
            processor.trip_generator()
        else:
            self.logger.info(f"Skipping trips generation for {processor_name}")
        # Generate demand data
        if "demand" in processor.getProcessingSteps():
            if (
                processor.getProcessorClass() == "GBFS"
                and not processor.getRotating()
                and "trips" not in processor.getProcessingSteps()
            ):
                self.logger.warning(
                    "Generation of demand for non-rotating GBFS processor without trips generation is not possible and will fail."
                )
            processor.demand_generator()
        else:
            self.logger.info(f"Skipping demand generation for {processor_name}")
        # Generate availability data
        if "availability" in processor.getProcessingSteps():
            processor.availability_generator()
        else:
            self.logger.info(f"Skipping availability generation for {processor_name}")
        self.logger.info(f"Completed {processor_name} processor")

    def run_extension(self, extension, extension_name):
        """Run single extensions."""
        # Single point of entry
        extension.run()

        self.logger.info(f"Completed {extension_name} extension")

    def process_operators(self):
        """Process all configured operators."""
        self.logger.info(
            f"Found {len(self.processors_config)} processors to process with {len(self.extension_config)} extensions"
        )

        # Process each configured processor
        for processor_item in self.processors_config:
            # Extract processor name and configuration
            # Each processor_item is a dict with one key (processor name) and value (config)
            processor_name = list(processor_item.keys())[0]
            processor_config = processor_item[processor_name]
            processor_steps = processor_config.get("steps")
            if processor_steps is None:
                processor_steps = ["trips", "availability", "demand"]
                self.logger.info(
                    f"No processing_steps specified for {processor_name}, defaulting to {processor_steps}"
                )
            else:
                self.logger.info(
                    f"Processing steps for {processor_name}: {processor_steps}"
                )

            processor_class = processor_config.get("class", "unknown")
            self.logger.info(f"Creating {processor_class} processor: {processor_name}")

            try:
                processor = self.create_processor(
                    processor_name,
                    processor_config,
                    processor_steps,
                )
                self.logger.info(f"Processing {processor_name}")

                # Run all generators
                self.run_generators(processor, processor_name)

            except Exception as e:
                self.logger.error(f"Error processing {processor_name}: {str(e)}")
                continue

        self.logger.info("All processors completed!")

    def process_extensions(self):
        """Process all configured extensions."""
        # Process each configured extension
        for extension_item in self.extension_config:
            # Extract extension name and configuration
            extension_name = list(extension_item.keys())[0]
            extension_config = extension_item[extension_name]

            extension_class = extension_config.get("class", "unknown")
            self.logger.info(f"Creating {extension_class} extension: {extension_name}")

            try:
                # Create extension instance
                extension = self.create_extension(extension_name, extension_config)

                # Run single extension
                self.run_extension(extension, extension_name)

            except Exception as e:
                self.logger.error(f"Error processing {extension_name}: {str(e)}")
                continue

        self.logger.info("All extensions completed!")

    def run(self):
        """Main method to process all configured operators and extensions."""
        self.process_transformers()
        self.process_mergers()
        self.process_operators()
        self.process_extensions()


def main():
    """Main function to create and run the DataPipelineManager."""
    pipeline_manager = DataPipelineManager()
    pipeline_manager.run()


if __name__ == "__main__":
    main()
