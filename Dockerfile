FROM continuumio/miniconda3

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy environment file
COPY environment.yml .

# Create and activate conda environment
RUN conda env create -f environment.yml

# Use conda shell for all RUN commands going forward
SHELL ["conda", "run", "-n", "genre-lyrics", "/bin/bash", "-c"]

# Copy the rest of the project
COPY . .

EXPOSE 8502

# Run the app with the environment
CMD ["conda", "run", "--no-capture-output", "-n", "appenv", "streamlit", "run", "src/app.py", "--server.port=8502", "--server.address=0.0.0.0"]
