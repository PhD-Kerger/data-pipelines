FROM python:3.12.12-trixie

# Set working directory
WORKDIR /app

# Install system dependencies (Debian)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    postgresql-client build-essential cmake libgdal-dev curl yq \
    && rm -rf /var/lib/apt/lists/*

# Install DuckDB CLI
RUN curl https://install.duckdb.org | sh \
    && ln -sf /root/.duckdb/cli/latest/duckdb /usr/local/bin/duckdb

# Install yq from GitHub releases
RUN wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/bin/yq \
    && chmod +x /usr/bin/yq

# Create Python virtual environment with requirements.txt
COPY requirements.txt .
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install -r requirements.txt

RUN rm requirements.txt

CMD ["tail", "-f", "/dev/null"]