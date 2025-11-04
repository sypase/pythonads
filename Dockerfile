# Use Python 3.9.6 image from Docker Hub
FROM python:3.9.6

# Set the working directory in the container
WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port that FastAPI runs on
EXPOSE 7777

# Run the FastAPI application with increased timeout and limits for large file uploads
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7777", "--timeout-keep-alive", "600", "--timeout-graceful-shutdown", "30", "--limit-concurrency", "1000", "--backlog", "2048", "--limit-max-requests", "10000", "--log-level", "info"]
