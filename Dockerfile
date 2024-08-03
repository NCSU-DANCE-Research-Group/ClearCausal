# Use an official Ubuntu image as the base image
FROM ubuntu:latest

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    libhdf5-dev \
    vim \
    && apt-get clean

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
ENV PATH="/root/.local/bin:$PATH"

# Set HDF5_DIR environment variable
ENV HDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/serial/

# Copy the project files to the container
WORKDIR /app
COPY . /app

# Install the dependencies using Poetry
RUN poetry install

# Set the default shell to bash
SHELL ["/bin/bash", "-c"]

# Activate the virtual environment and run a command
CMD ["/bin/bash", "-c", "source $(poetry env info --path)/bin/activate && exec bash"]
