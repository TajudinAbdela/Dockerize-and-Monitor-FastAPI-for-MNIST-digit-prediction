# Use a base image with Python installed
FROM python:3.12
# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn prometheus-fastapi-instrumentator

# Copy the FastAPI application code
COPY . /app
# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI application code to the container
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Define the command to run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]





