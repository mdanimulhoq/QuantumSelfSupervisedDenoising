# N2LN-QEM Dockerfile (TDD §10.3)
# Usage: docker build -t n2ln-qem .
#        docker run -it n2ln-qem make reproduce

FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install package
RUN pip install -e .

# Set default command
CMD ["make", "help"]
