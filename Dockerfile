# Use the Python slim image as the base
FROM python:3.12-slim

# Install system dependencies required for GDAL and rasterio
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    && apt-get clean

# Set the GDAL_CONFIG environment variable for rasterio
ENV GDAL_CONFIG=/usr/bin/gdal-config
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt to the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Set the command to run the application
CMD ["python", "app.py"]