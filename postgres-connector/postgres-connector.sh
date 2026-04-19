#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="env.yaml"

# === INPUT PARAMETERS ===
if [ $# -lt 1 ]; then
  echo "Usage: $0 <datatype> [operator(s)]"
  echo "Supported data types:"
  echo "  - demand, availability, trips (require operators)"
  echo "  - weather, osm, osm_landuse, foursquare, holidays, gtfs, wfs, demographics_MA (standalone), bike_count_stations"
  echo "  - geo (metadata only - also auto-updated when needed)"
  echo ""
  echo "Note: geo metadata (geo_information, station_names) is automatically"
  echo "      updated when processing any datatype that has location_id references."
  echo ""
  echo "Examples:"
  echo "  $0 geo                    # Update only geo metadata"
  echo "  $0 weather                # Process weather data (geo auto-updated)"
  echo "  $0 demand tier            # Process demand for tier operator"
  echo "  $0 demand tier,dott       # Process demand for multiple operators"
  echo "  $0 trips tier,dott        # Process trips for multiple operators"
  exit 1
fi

DATATYPE="$1"
OPERATORS=()

# Supported data types and their corresponding table names
declare -A TABLE_MAP=(
  [trips]=trips
  [demand]=demand
  [availability]=availability
  [weather]=weather
  [osm]=osm
  [osm_landuse]=osm_landuse
  [foursquare]=foursquare
  [holidays]=holidays
  [gtfs]=gtfs
  [wfs]=wfs
  [demographics_MA]=demographics
  [bike_count_stations]=bike_counting_stations
)

# Data types that require operators
OPERATOR_REQUIRED_TYPES=("demand" "availability" "trips")

# Data types that require a second parameter (like region name)
SECOND_PARAMETER_REQUIRED_TYPES=("bike_count_stations")

# Data types that are metadata (don't follow operator/datatype structure)
METADATA_TYPES=("geo")

# Data types that require geo metadata (have location_id foreign key)
GEO_DEPENDENT_TYPES=("demand" "availability" "trips" "weather" "osm" "foursquare" "gtfs" "bike_count_stations")

# Data types that should NOT be cleared before inserting
NO_CLEAR_TYPES=("gtfs" "demand" "availability" "trips")

# Check if datatype is supported
if [[ -z "${TABLE_MAP[$DATATYPE]:-}" ]] && [[ ! " ${METADATA_TYPES[@]} " =~ " $DATATYPE " ]]; then
  echo "Error: Unsupported datatype '$DATATYPE'"
  echo "Supported: ${!TABLE_MAP[@]} ${METADATA_TYPES[@]}"
  exit 1
fi

# Check if operators are required for this datatype
if [[ " ${OPERATOR_REQUIRED_TYPES[@]} " =~ " $DATATYPE " ]]; then
  if [ $# -ne 2 ]; then
    echo "Error: Datatype '$DATATYPE' requires operators"
    echo "Usage: $0 $DATATYPE <operator(s)>"
    echo "Example: $0 $DATATYPE tier,dott"
    exit 1
  fi
  # Split comma-separated operators into array
  IFS=',' read -ra OPERATORS <<< "$2"
elif [[ " ${SECOND_PARAMETER_REQUIRED_TYPES[@]} " =~ " $DATATYPE " ]]; then
  if [ $# -ne 2 ]; then
    echo "Error: Datatype '$DATATYPE' requires a second parameter"
    echo "Usage: $0 $DATATYPE <region_name>"
    echo "Example: $0 $DATATYPE berlin"
    exit 1
  fi
  ADDITIONAL_PARAMETER="$2"
elif [ $# -gt 1 ]; then
  echo "Warning: Datatype '$DATATYPE' does not use operators. Ignoring operator parameter."
fi

# === READ DIRECTORIES FROM YAML ===
INPUT_DIR=$(yq -r '.processing.directories.input' "$ENV_FILE")
OUTPUT_DIR=$(yq -r '.processing.directories.output' "$ENV_FILE")
METADATA_DIR=$(yq -r '.processing.directories.metadata' "$ENV_FILE")
INTERNAL_DIR=$(yq -r '.processing.directories.internal' "$ENV_FILE")
EXTENSION_DIR=$(yq -r '.processing.directories.extensions' "$ENV_FILE")

# === READ POSTGRES CONNECTION FROM YAML ===
PG_DB=$(yq -r '.processing.postgres.dbname' "$ENV_FILE")
PG_USER=$(yq -r '.processing.postgres.user' "$ENV_FILE")
PG_PASS=$(yq -r '.processing.postgres.password' "$ENV_FILE")
PG_HOST=$(yq -r '.processing.postgres.host' "$ENV_FILE")
PG_PORT=$(yq -r '.processing.postgres.port' "$ENV_FILE")

PG_CONN="dbname=$PG_DB user=$PG_USER password=$PG_PASS host=$PG_HOST port=$PG_PORT"

# Logging function for professional log output
log() {
  local level="$1"
  shift
  local msg="$*"
  local ts
  ts=$(date '+%Y-%m-%d %H:%M:%S')
  local script_name
  script_name=$(basename "$0")
  echo "$ts - $script_name - $level - $msg"
}

# === HANDLE METADATA UPDATES ===
update_metadata() {
  log INFO "Updating metadata tables geo_information and station_names..."
  
  # Temporarily disable foreign key constraints to allow metadata updates
  PGPASSWORD="$PG_PASS" psql -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d "$PG_DB" -c "
    -- Disable all triggers (including foreign key constraint triggers)
    ALTER TABLE geo_information DISABLE TRIGGER ALL;
    ALTER TABLE station_names DISABLE TRIGGER ALL;

    DELETE FROM geo_information;
    DELETE FROM station_names;
  "
  
  # Insert new data from parquet files
  duckdb -c "
    SET threads TO $(nproc);
    SET memory_limit = '$(free -g | awk '/^Mem:/{print int($2*0.8)}')GB';
    INSTALL postgres;
    LOAD postgres;
    ATTACH '$PG_CONN' AS pg (TYPE POSTGRES);
    
    INSERT INTO pg.geo_information FROM '$EXTENSION_DIR/geo/geo_information.parquet';
    INSERT INTO pg.station_names FROM '$METADATA_DIR/station_names.parquet';
    "
  
  # Re-enable foreign key constraints and flip coordinates
  PGPASSWORD="$PG_PASS" psql -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d "$PG_DB" -c "
    -- Flip coordinates in geo_information location column
    ALTER TABLE geo_information 
      ALTER COLUMN location 
        TYPE geometry(Point,4326)
        USING ST_FlipCoordinates(location::geometry);

    ALTER TABLE geo_information
      ALTER COLUMN location
        TYPE geography(Point,4326)
        USING location::geography;


    -- Re-enable all triggers (including foreign key constraint triggers)
    ALTER TABLE geo_information ENABLE TRIGGER ALL;
    ALTER TABLE station_names ENABLE TRIGGER ALL;
  "
  
  log INFO "Metadata tables updated."
}

# === PROCESS DATA BASED ON TYPE ===
process_datatype() {
  local datatype="$1"
  local target_table="${TABLE_MAP[$datatype]}"
  
  log INFO "Processing datatype: $datatype"
  log INFO "Target table: $target_table"
  
  # Check if geo metadata update is needed for this datatype
  if [[ " ${GEO_DEPENDENT_TYPES[@]} " =~ " $datatype " ]]; then
    log INFO "Datatype '$datatype' requires geo metadata. Updating geo_information and station_names..."
    update_metadata
  fi
  
  case "$datatype" in
    "demand"|"availability"|"trips")
      log INFO "Processing operator-based datatype: $datatype"
      process_operator_based_data "$datatype" "$target_table"
      ;;
    "weather"|"osm"|"osm_landuse"|"foursquare"|"holidays"|"gtfs"|"wfs"|"demographics_MA"|"bike_count_stations")
      process_standalone_data "$datatype" "$target_table"
      ;;
  esac
}

# === PROCESS OPERATOR-BASED DATA ===
process_operator_based_data() {
  local datatype="$1"
  local target_table="$2"
  
  for operator in "${OPERATORS[@]}"; do
    local data_path="$OUTPUT_DIR/$operator/$datatype"
    
    if [ ! -d "$data_path" ]; then
      log ERROR "Directory not found: $data_path"
      continue
    fi
    
    log INFO "Processing operator: $operator"
    log INFO "Data path: $data_path"
    
    process_parquet_files "$data_path" "$target_table"
  done
}

# === PROCESS STANDALONE DATA ===
process_standalone_data() {
  local datatype="$1"
  local target_table="$2"
  
  case "$datatype" in
    "weather")
      local data_path="$EXTENSION_DIR/$datatype"
      ;;
    "osm")
      local data_path="$EXTENSION_DIR/$datatype"
      ;;
    "osm_landuse")
      local data_path="$EXTENSION_DIR/$datatype"
      ;;
    "foursquare")
      local data_path="$EXTENSION_DIR/$datatype"
      ;;
    "holidays")
      local data_path="$EXTENSION_DIR/$datatype"
      ;;
    "gtfs")
      local data_path="$EXTENSION_DIR/$datatype"
      ;;
    "wfs")
      local data_path="$EXTENSION_DIR/$datatype"
      ;;
    "demographics_MA")
      local data_path="$EXTENSION_DIR/$datatype"
      ;;
    "bike_count_stations")
      if [ $# -lt 2 ]; then
        log ERROR "Datatype 'bike_count_stations' requires region name"
        echo "Usage: $0 bike_count_stations <region_name>"
        exit 1
      fi
      local data_path="$EXTENSION_DIR/$datatype/$ADDITIONAL_PARAMETER"
      log INFO "Processing bike_count_stations for region: $ADDITIONAL_PARAMETER"
      log INFO "Data path: $data_path"
      ;;
    *)
      log ERROR "Unknown standalone datatype: $datatype"
      return 1
      ;;
  esac
  
  if [ ! -d "$data_path" ]; then
    log ERROR "Directory not found: $data_path"
    return 1
  fi
  
  # Clear the table before inserting new data (unless it's in NO_CLEAR_TYPES)
  if [[ ! " ${NO_CLEAR_TYPES[@]} " =~ " $datatype " ]]; then
    log INFO "Clearing table $target_table before inserting new data..."
    PGPASSWORD="$PG_PASS" psql -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d "$PG_DB" -c "DELETE FROM $target_table;"
    log INFO "Table $target_table cleared."
  else
    log INFO "Skipping table clearing for $target_table (configured to preserve existing data)"
  fi
  
  log INFO "Data path: $data_path"
  process_parquet_files "$data_path" "$target_table"
}

# === PROCESS PARQUET FILES ===
process_parquet_files() {
  local data_path="$1"
  local target_table="$2"
  
  local files_found=false
  for f in "$data_path"/*.parquet; do
    [ -e "$f" ] || continue
    files_found=true
    
    log INFO "Loading $f into $target_table..."

    duckdb -c "
          SET threads TO $(nproc);
          SET memory_limit = '$(free -g | awk '/^Mem:/{print int($2*0.8)}')GB';
          INSTALL postgres;
          LOAD postgres;
          INSTALL spatial;
          LOAD spatial;
          ATTACH '$PG_CONN' AS pg (TYPE POSTGRES);
          INSERT INTO pg.$target_table FROM '$f';
    "
    log INFO "Finished loading $f"
  done
  
  if [ "$files_found" = false ]; then
    log WARNING "No parquet files found in $data_path"
  fi
}

# === MAIN EXECUTION ===
case "$DATATYPE" in
  "geo")
    log INFO "Processing metadata (geo_information and station_names)..."
    update_metadata
    ;;
  *)
    process_datatype "$DATATYPE"
    ;;
esac

log INFO "Processing completed for datatype: $DATATYPE"
