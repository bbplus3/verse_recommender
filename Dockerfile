# Use official Python image as base
FROM python:3.11

# Set working directory in the container
WORKDIR /verse_recommender

# Copy requirements.txt before installing dependencies (for caching efficiency)
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the Streamlit app
ENTRYPOINT ["streamlit", "run", "st_bible.py", "--server.port=8501", "--server.address=0.0.0.0"]
