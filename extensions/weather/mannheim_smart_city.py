import datetime
import os
from pathlib import Path
import shutil
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as csv
import pyarrow.parquet as pq
import pyarrow_ops as pa_ops
import requests
import json
from bs4 import BeautifulSoup
import pdfplumber

import pytz

from utils.data_pipeline_logger import DataPipelineLogger

class MannheimSmartCityWeatherExtension:
    """Extension for Mannheim weather stations from smartmannheim.de."""

    def __init__(
        self,
        extension_data_dir_path,
        meta_data_dir_path,
        input_data_dir_path,
        logs_data_dir_path,
        start_date,
        end_date,
        locations,
    ):
        self.extension_data_dir_path = extension_data_dir_path + "/weather"
        self.meta_data_dir_path = meta_data_dir_path
        self.log_file = Path(logs_data_dir_path) / "logs.log"
        self.extension_temp_dir_path = extension_data_dir_path + "/tmp"

        # Setup logger
        self.logger = DataPipelineLogger.get_logger(
            name=self.__class__.__name__, log_file_path=self.log_file
        )

        os.makedirs(self.extension_data_dir_path, exist_ok=True)
        os.makedirs(self.extension_temp_dir_path, exist_ok=True)

        # start and end date are in format YYYYMMDD. they need to be converted to yyyy-mm-dd
        self.start_date = start_date
        self.end_date = end_date
        
    @property
    def start_timestamp(self):
        """Convert start_date to timestamp once and cache it"""
        if not hasattr(self, "_start_timestamp"):
            self._start_timestamp = int(
                datetime.datetime.strptime(self.start_date, "%Y%m%d")
                .replace(tzinfo=pytz.UTC)
                .timestamp()
            )
        return self._start_timestamp

    @property
    def end_timestamp(self):
        """Convert end_date to timestamp once and cache it"""
        if not hasattr(self, "_end_timestamp"):
            self._end_timestamp = int(
                datetime.datetime.strptime(self.end_date, "%Y%m%d")
                .replace(tzinfo=pytz.UTC)
                .timestamp()
            )
        return self._end_timestamp

    def _wind_direction_to_int64(self, values):
        """Normalize wind direction values to nullable int64 using round-half-up."""
        result = []
        for value in values:
            if value is None:
                result.append(None)
                continue
            try:
                result.append(int(round(float(value))))
            except (TypeError, ValueError):
                result.append(None)
        return pa.array(result, type=pa.int64())
    
    def run(self):
        """Run the extension to fetch and process weather data."""
        self.logger.info("Starting Mannheim Smart City Weather Extension")
        self.fetch_station_metadata()
        self.fetch_weather_data()
        self.process_weather_data()
        self.logger.info("Finished Mannheim Smart City Weather Extension")
        
    def fetch_station_metadata(self):
        """Fetch station metadata from PDF document."""
        pdf_url = "https://smartmannheim.de/wp-content/uploads/2024/07/20240314_Metadatenkatalog_MA_Klimamessnetz.pdf"
        
        # Define path for cached metadata
        metadata_pdf_path = os.path.join(self.meta_data_dir_path, "mannheim_climate_stations.pdf")
        metadata_json_path = os.path.join(self.meta_data_dir_path, "mannheim_climate_stations.json")
        
        # Check if we already have cached metadata
        if os.path.exists(metadata_json_path):
            self.logger.info(f"Loading cached station metadata from {metadata_json_path}")
            with open(metadata_json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # Download PDF if not already cached
        if not os.path.exists(metadata_pdf_path):
            self.logger.info(f"Downloading station metadata PDF from {pdf_url}")
            try:
                response = requests.get(pdf_url, timeout=30)
                if response.status_code == 200:
                    os.makedirs(os.path.dirname(metadata_pdf_path), exist_ok=True)
                    with open(metadata_pdf_path, 'wb') as f:
                        f.write(response.content)
                    self.logger.info(f"Downloaded PDF to {metadata_pdf_path}")
                else:
                    self.logger.error(f"Failed to download PDF: HTTP {response.status_code}")
                    return None
            except Exception as e:
                self.logger.error(f"Error downloading PDF: {str(e)}")
                return None
        
        # Extract tables from pages 28-36
        self.logger.info(f"Extracting station metadata from PDF pages 28-36")
        stations = []
        
        try:
            with pdfplumber.open(metadata_pdf_path) as pdf:
                # Pages are 0-indexed in pdfplumber, so pages 28-36 are indices 27-35
                for page_num in range(27, 36):
                    if page_num >= len(pdf.pages):
                        break
                    
                    page = pdf.pages[page_num]
                    tables = page.extract_tables()
                    
                    for table in tables:
                        if not table:
                            continue
                        
                        # Try to identify header row and extract data
                        # Common metadata fields might include: Station ID, Name, Latitude, Longitude, etc.
                        for i, row in enumerate(table):
                            if not row or all(cell is None or str(cell).strip() == '' for cell in row):
                                continue
                            
                            # Skip header rows (typically containing field names)
                            if i == 0 or any(header in str(row).lower() for header in ['station', 'id', 'name', 'lat', 'lon']):
                                continue
                            
                            # Process data rows
                            # This is a generic parser - adjust based on actual PDF structure
                            # Filter out empty cells
                            filtered_row_data = [str(cell).strip() for cell in row if cell and str(cell).strip()]
                            
                            # Extract only specific indices: 2nd (index 1), 4th (index 3), 5th (index 4), 6th (index 5), 7th (index 6)
                            if len(filtered_row_data) >= 7:
                                station_data = {
                                    'nr': filtered_row_data[1].replace('/', ''),      # 2nd entry
                                    'id': filtered_row_data[3],                         # 4th entry
                                    'id_add': filtered_row_data[4].replace('/', ''),  # 5th entry
                                    'lat': round(float(filtered_row_data[5]), 3),     # 6th entry
                                    'lng': round(float(filtered_row_data[6]), 3)      # 7th entry
                                }
                            else:
                                # Skip rows that don't have enough data
                                continue
                            
                            stations.append(station_data)
                
                self.logger.info(f"Extracted {len(stations)} station records from PDF")
                
                # Save to JSON for caching
                if stations:
                    with open(metadata_json_path, 'w', encoding='utf-8') as f:
                        json.dump(stations, f, indent=2, ensure_ascii=False)
                    self.logger.info(f"Saved station metadata to {metadata_json_path}")
                
                return stations
                
        except Exception as e:
            self.logger.error(f"Error extracting data from PDF: {str(e)}")
            return None
        
    def fetch_weather_data(self):
        """Fetch weather data from Mannheim smart city API."""
        # we need to analyze html page to get the download links
        url = "https://opendata.smartmannheim.de/dataset/klimadaten-mannheim"
        response = requests.get(url)
        if response.status_code != 200:
            self.logger.error(f"Failed to fetch data from {url}")
            return
        html_content = response.text
        
        # under a ul with class="resource-list" we have multiple li elements with class="resource-item" that all have a data id. we need all data ids.
        soup = BeautifulSoup(html_content, 'html.parser')
        resource_list = soup.find('ul', class_='resource-list')
        resource_items = resource_list.find_all('li', class_='resource-item')
        data_ids = []
        for item in resource_items:
            data_id = item.get('data-id')
            # in the li element we have a a element title="YYYY-MM-DD.zip". check if the date is in range
            title_element = item.find('a', title=True)
            if title_element:
                title = title_element['title']
                date_str = title.split(".")[0]
                date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                date_timestamp = int(date_obj.replace(tzinfo=pytz.UTC).timestamp())
                if date_timestamp < self.start_timestamp or date_timestamp > self.end_timestamp:
                    self.logger.info(f"Skipping data id {data_id} as it is out of date range.")
                    continue
            
            if data_id:
                data_ids.append((data_id, date_str))
                
        self.logger.info(f"Found {len(data_ids)} data ids.")
        
        # now we can construct the download links (https://opendata.smartmannheim.de/dataset/klimadaten-mannheim/resource/{data_id})
        download_links = [
            f"https://opendata.smartmannheim.de/dataset/klimadaten-mannheim/resource/{data_id}/download/{date_str}.zip"
            for data_id, date_str in data_ids
        ]
        self.logger.info(f"Constructed {len(download_links)} download links.")
        # download all files to temp directory
        for link in download_links:
            file_name = link.split("/")[-1]
            temp_file_path = os.path.join(self.extension_temp_dir_path, file_name)
            self.logger.info(f"Downloading {link} to {temp_file_path}")
            response = requests.get(link, stream=True)
            if response.status_code == 200:
                with open(temp_file_path, 'wb') as f:
                    shutil.copyfileobj(response.raw, f)
                self.logger.info(f"Downloaded {file_name} successfully.")
            else:
                self.logger.error(f"Failed to download {link}")
                
        # unzip all files in temp directory in seperate folders
        for file in os.listdir(self.extension_temp_dir_path):
            if file.endswith(".zip"):
                file_path = os.path.join(self.extension_temp_dir_path, file)
                self.logger.info(f"Unzipping {file_path}")
                unzip_dir = os.path.join(self.extension_temp_dir_path, file.replace(".zip", ""))
                shutil.unpack_archive(file_path, unzip_dir)
                self.logger.info(f"Unzipped {file_path} successfully.")
    
    def _get_unique_location_coordinates(self, all_tables):
        """Extract unique coordinates and append to location_coordinates.parquet"""
        try:
            # Define the meta data file path first
            meta_dir = Path(self.meta_data_dir_path)
            meta_dir.mkdir(parents=True, exist_ok=True)
            coordinates_file = meta_dir / "location_coordinates.parquet"

            # Get unique coordinates from all tables
            unique_coords = set()
            for table, lat, lng in all_tables:
                unique_coords.add((lat, lng))

            if not unique_coords:
                self.logger.warning("No coordinates found in weather data")
                return

            # Convert to list for processing
            new_coords = list(unique_coords)

            # Handle existing file
            if coordinates_file.exists():
                # Read existing data
                existing_data = pq.read_table(coordinates_file)
                existing_coords = set(
                    zip(
                        existing_data.column("lat").to_pylist(),
                        existing_data.column("lng").to_pylist(),
                    )
                )
                max_location_id = max(existing_data.column("location_id").to_pylist())

                # Filter out coordinates that already exist
                new_coords = [
                    (lat, lng)
                    for lat, lng in new_coords
                    if (lat, lng) not in existing_coords
                ]

                if not new_coords:
                    self.logger.info("No new coordinates to add")
                    return

                self.logger.info(
                    f"Adding {len(new_coords)} new coordinates to existing file"
                )
            else:
                max_location_id = 0
                self.logger.info(
                    f"Creating new coordinates file with {len(new_coords)} coordinates"
                )

            # Create data for new coordinates only
            if new_coords:
                new_lats, new_lngs = zip(*new_coords)
                new_location_ids = [
                    max_location_id + i + 1 for i in range(len(new_coords))
                ]

                new_data = pa.table(
                    {"location_id": new_location_ids, "lat": new_lats, "lng": new_lngs}
                )

                # Append to existing file or create new one
                if coordinates_file.exists():
                    existing_data = pq.read_table(coordinates_file)
                    combined_data = pa.concat_tables([existing_data, new_data])
                    pq.write_table(
                        combined_data, coordinates_file, compression="BROTLI"
                    )
                else:
                    pq.write_table(new_data, coordinates_file, compression="BROTLI")

                self.logger.info(f"Saved coordinates to {coordinates_file}")

        except Exception as e:
            self.logger.error(f"Error extracting coordinates: {e}")

    def _map_coordinates_to_location_ids(self):
        """Map coordinates to location IDs from the parquet file"""
        try:
            # Read the location coordinates file
            meta_dir = Path(self.meta_data_dir_path)
            coordinates_file = meta_dir / "location_coordinates.parquet"

            if not coordinates_file.exists():
                self.logger.error(
                    "location_coordinates.parquet not found. Run _get_unique_location_coordinates first."
                )
                return {}

            # Read the parquet file and create a mapping
            location_data = pq.read_table(coordinates_file)
            coord_to_id = {}

            location_ids = location_data.column("location_id").to_pylist()
            lats = location_data.column("lat").to_pylist()
            lngs = location_data.column("lng").to_pylist()

            for location_id, lat, lng in zip(location_ids, lats, lngs):
                coord_to_id[(round(lat, 3), round(lng, 3))] = location_id

            return coord_to_id

        except Exception as e:
            self.logger.error(f"Error mapping coordinates to location IDs: {e}")
            return {}
                
    def process_weather_data(self):
        station_metadata_json = os.path.join(self.meta_data_dir_path, "mannheim_climate_stations.json")
        if not os.path.exists(station_metadata_json):
            self.logger.error("Station metadata JSON not found. Cannot process weather data.")
            return
        
        with open(station_metadata_json, 'r', encoding='utf-8') as f:
            station_metadata = json.load(f)
        
        # Collect all tables to combine them later
        all_tables = []
        
        for folder in os.listdir(self.extension_temp_dir_path):
            folder_path = os.path.join(self.extension_temp_dir_path, folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith(".csv"):
                        file_path = os.path.join(folder_path, file)
                        self.logger.info(f"Processing {file_path}")
                        # read csv with pyarrow
                        table = csv.read_csv(file_path, parse_options=csv.ParseOptions(delimiter=','))
                        
                        # Handle duplicate columns by keeping only the first occurrence
                        seen_columns = {}
                        unique_indices = []
                        for i, col_name in enumerate(table.column_names):
                            if col_name not in seen_columns:
                                seen_columns[col_name] = i
                                unique_indices.append(i)
                        
                        if len(unique_indices) < len(table.column_names):
                            self.logger.warning(f"Found duplicate columns in {file}. Keeping only first occurrence.")
                            table = pa.table({table.column_names[i]: table.column(i) for i in unique_indices})
                        
                        # file name is {nr}-{id}-{id_add}.csv
                        file_name_parts = file.replace(".csv", "").split("-")
                        if len(file_name_parts) != 3:
                            self.logger.warning(f"Unexpected file name format: {file}. Skipping.")
                            continue
                        nr, id_, id_add = file_name_parts
                        # find station metadata
                        station_info = next((station for station in station_metadata if station['nr'] == nr and station['id'] == id_ and station['id_add'] == id_add), None)
                        if not station_info:
                            self.logger.warning(f"No metadata found for station {nr}-{id_}-{id_add}. Skipping.")
                            continue
                        
                        # Define standard columns we want to keep
                        standard_columns = ['timestamps', 'airHumidity', 'temperature', 'averageWindDirection', 'minWindSpeed']
                        
                        # Create a dictionary to hold column data
                        table_dict = {}
                        
                        # Add existing columns and cast appropriately
                        for col in standard_columns:
                            if col in table.column_names:
                                if col == 'timestamps':
                                    # print first 5 entries of timestamp column
                                    table_dict["timestamp"] = pc.cast(table.column(col), pa.timestamp('ns', tz='UTC'))
                                elif col == 'averageWindDirection':
                                    table_dict[col] = self._wind_direction_to_int64(table.column(col).to_pylist())
                                else:
                                    # Cast numeric columns to float64
                                    table_dict[col] = pc.cast(table.column(col), pa.float64())
                            else:
                                if col == 'averageWindDirection':
                                    null_array = pa.array([None] * table.num_rows, type=pa.int64())
                                else:
                                    null_array = pa.array([None] * table.num_rows, type=pa.float64())
                                table_dict[col] = null_array
                        
                        # Add metadata columns
                        table_dict['lat'] = pa.array([float(station_info['lat'])] * table.num_rows, type=pa.float64())
                        table_dict['lng'] = pa.array([float(station_info['lng'])] * table.num_rows, type=pa.float64())
                        
                        # Create table with consistent schema
                        table = pa.table(table_dict)
                        
                        all_tables.append((table, station_info['lat'], station_info['lng']))
                        self.logger.info(f"Added {table.num_rows} rows from station {nr}-{id_}-{id_add}")
        
        # Group tables by (lat, lng) and merge them
        if all_tables:
            self.logger.info(f"Grouping and merging {len(all_tables)} tables by location...")
            
            # First, ensure location coordinates are up to date
            self._get_unique_location_coordinates(all_tables)
            
            # Get coordinate to location ID mapping
            coord_to_id = self._map_coordinates_to_location_ids()
            if not coord_to_id:
                self.logger.error("No coordinate mappings available")
                return
            
            # Group tables by (lat, lng)
            from collections import defaultdict
            grouped_tables = defaultdict(list)
            for table, lat, lng in all_tables:
                grouped_tables[(lat, lng)].append(table)
            
            self.logger.info(f"Found {len(grouped_tables)} unique locations")
            
            merged_tables = []
            for (lat, lng), tables in grouped_tables.items():
                if len(tables) == 1:
                    merged_tables.append((tables[0], lat, lng))
                else:
                    self.logger.info(f"Merging {len(tables)} tables for location ({lat}, {lng})")
                    # Convert to pandas for easier merging
                    import pandas as pd
                    dfs = [t.to_pandas() for t in tables]
                    
                    # Start with the first dataframe
                    merged_df = dfs[0]
                    
                    # Merge with remaining dataframes on timestamp, lat, lng
                    for df in dfs[1:]:
                        merged_df = pd.merge(
                            merged_df, df,
                            on=['timestamp', 'lat', 'lng'],
                            how='outer',
                            suffixes=('', '_dup')
                        )
                        
                        # Handle duplicate columns by coalescing (taking first non-null value)
                        for col in ['airHumidity', 'temperature', 'averageWindDirection', 'minWindSpeed']:
                            if f'{col}_dup' in merged_df.columns:
                                merged_df[col] = merged_df[col].fillna(merged_df[f'{col}_dup'])
                                merged_df = merged_df.drop(columns=[f'{col}_dup'])

                    # Ensure wind direction remains nullable integer before Arrow conversion.
                    if 'averageWindDirection' in merged_df.columns:
                        normalized_wind = self._wind_direction_to_int64(
                            merged_df['averageWindDirection'].tolist()
                        ).to_pylist()
                        merged_df['averageWindDirection'] = pd.Series(
                            normalized_wind, dtype='Int64'
                        )
                    
                    # Remove duplicate rows based on timestamp, lat, lng
                    merged_df = merged_df.drop_duplicates(subset=['timestamp', 'lat', 'lng'])
                    
                    # Convert back to pyarrow table
                    schema = pa.schema([
                        ('timestamp', pa.timestamp('ns', tz='UTC')),
                        ('airHumidity', pa.float64()),
                        ('temperature', pa.float64()),
                        ('averageWindDirection', pa.int64()),
                        ('minWindSpeed', pa.float64()),
                        ('lat', pa.float64()),
                        ('lng', pa.float64())
                    ])
                    merged_table = pa.Table.from_pandas(merged_df, schema=schema, preserve_index=False)
                    merged_tables.append((merged_table, lat, lng))
                    self.logger.info(f"Merged table has {merged_table.num_rows} rows for location ({lat}, {lng})")
            
            # Now convert to final schema with location_ids
            final_tables = []
            for table, lat, lng in merged_tables:
                location_id = coord_to_id.get((round(lat, 3), round(lng, 3)))
                if location_id is None:
                    self.logger.warning(f"No location_id found for ({lat}, {lng})")
                    continue

                # Convert to dict for direct PyArrow table creation (avoids pandas metadata)
                table_dict = {}
                
                # Get all columns as Python lists
                for col_name in table.column_names:
                    table_dict[col_name] = table.column(col_name).to_pylist()
                
                # Add location_id
                num_rows = table.num_rows
                table_dict["location_id"] = [location_id] * num_rows
                
                # Add precipitation column (all nulls)
                table_dict["precipitation"] = [None] * num_rows
                
                # Rename columns and prepare final dict
                final_dict = {
                    "location_id": table_dict["location_id"],
                    "timestamp": table_dict["timestamp"],
                    "temperature": table_dict.get("temperature", [None] * num_rows),
                    "humidity": table_dict.get("airHumidity", [None] * num_rows),
                    "precipitation": table_dict["precipitation"],
                    "wind_speed": table_dict.get("minWindSpeed", [None] * num_rows),
                    "wind_direction": table_dict.get("averageWindDirection", [None] * num_rows),
                }
                
                # Convert to pyarrow with explicit schema
                final_schema = pa.schema(
                    [
                        ("location_id", pa.int32()),
                        ("timestamp", pa.timestamp("ns", tz="UTC")),
                        ("temperature", pa.float64()),
                        ("humidity", pa.float64()),
                        ("precipitation", pa.float64()),
                        ("wind_speed", pa.float64()),
                        ("wind_direction", pa.int64()),
                    ]
                )
                
                # Create arrays with proper types
                arrays = [
                    pa.array(final_dict["location_id"], type=pa.int32()),
                    pa.array(final_dict["timestamp"], type=pa.timestamp("ns", tz="UTC")),
                    pa.array(final_dict["temperature"], type=pa.float64()),
                    pa.array(final_dict["humidity"], type=pa.float64()),
                    pa.array(final_dict["precipitation"], type=pa.float64()),
                    pa.array(final_dict["wind_speed"], type=pa.float64()),
                    self._wind_direction_to_int64(final_dict["wind_direction"]),
                ]
                
                final_table = pa.table(arrays, schema=final_schema)
                final_tables.append(final_table)
            
            # Combine all final tables
            self.logger.info(f"Combining {len(final_tables)} final tables into one...")
            combined_table = pa.concat_tables(final_tables)
            
            # Deduplicate and sort
            combined_table = pa_ops.drop_duplicates(
                combined_table,
                ["location_id", "timestamp"],
                keep="first",
            )
            combined_table = combined_table.sort_by(
                [("location_id", "ascending"), ("timestamp", "ascending")]
            )
            
            self.logger.info(f"Combined table has {combined_table.num_rows} total rows")
            
            # Write to parquet file
            output_parquet = os.path.join(self.extension_data_dir_path, "mannheim_smart_city_weather.parquet")
            # Delete existing file if it exists
            if os.path.exists(output_parquet):
                os.remove(output_parquet)
            pq.write_table(combined_table, output_parquet, compression="BROTLI")
            self.logger.info(f"Saved combined weather data to {output_parquet}")
            
            # Clean up temp directory
            if os.path.exists(self.extension_temp_dir_path):
                shutil.rmtree(self.extension_temp_dir_path)
                self.logger.info("Cleaned up temporary directory")
        else:
            self.logger.warning("No tables to combine.")