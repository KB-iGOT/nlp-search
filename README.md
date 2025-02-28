# nlp-search

python version required: 3.10


# Installation

    $pip install -r requirements.txt

# Configuration

- in the .env file update following parameters

    project
    location
    GOOGLE_APPLICATION_CREDENTIALS
- Add the google service credential file in creds folder.

# Running the service

    $uvicorn --reload src.main:app

# Integration

- Access the APIs at http://<server>:8000/docs

# Running with Docker

    $docker build -t nlp-search .
    $docker run -p 8000:8000  nlp-search:latest

