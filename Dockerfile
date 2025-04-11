# Use official Python image as base
FROM python:3.11

# Set working directory in the container
WORKDIR /verse_recommender

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/bbplus3/verse_recommender.git .

# Copy requirements.txt before installing dependencies (for caching efficiency)
# COPY requirements.txt requirements.txt
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "st_bible.py", "--server.port=8501", "--server.address=0.0.0.0"]
# Run the Streamlit app
# CMD ["streamlit", "run", "st_bible.py", "--server.port=8501", "--server.address=0.0.0.0"]
