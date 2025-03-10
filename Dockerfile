# Use the official Python image from the Docker Hub
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Specify the port number the container should expose
EXPOSE 7777

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7777"]