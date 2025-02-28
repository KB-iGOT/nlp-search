# Use the official Python image from the Docker Hub
FROM python:3.10.16-bullseye

# Set the working directory in the container
WORKDIR .

# Copy the entire application code into the container
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Expose the port that the FastAPI app will run on
EXPOSE 8000

# Command to run the FastAPI application using Uvicorn
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]