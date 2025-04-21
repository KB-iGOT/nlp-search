# NLP Search Service

A fast, configurable NLP search service powered by large language models (LLMs).

---

## Description

**NLP Search** is an expert natural language processing service designed to extract the most relevant and meaningful keywords from user-provided text. Leveraging advanced large language models (LLMs), the service focuses on identifying important concepts, entities, topics, and unique terms—while filtering out generic stopwords and overly common words. smaking it easy to integrate with downstream applications or search systems.

---

## Prerequisites

- **Python 3.11+**:  Ensure you have Python 3.11 or a later version installed.
- **Google Cloud Project**: Required for Vertex AI integration and credential management.
- **UV**: Used for dependency and environment management. [Install uv by following the official instructions.](https://docs.astral.sh/uv/getting-started/installation/#pypi)
- **Git**: Ensure you have Git installed for version control and project setup.



## Setup

### 1. Clone the Repository

Open your terminal and run:

```bash
git clone https://github.com/KB-iGOT/nlp-search.git
cd nlp-search
```

### 2. Create and Activate a Virtual Environment

It is recommended to use a virtual environment to manage dependencies:

```bash
python -m venv venv
source venv/bin/activate    # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install uv && uv pip install -r requirements.txt
```

---

## Configuration

1. **Set up Google Cloud credentials:**  
   - Ensure you have a Google Cloud project.
   - Make sure you have the Vertex AI API enabled in your project.
   - Set the `GOOGLE_CLOUD_LOCATION`, `GOOGLE_CLOUD_PROJECT`, and `GOOGLE_APPLICATION_CREDENTIALS` environment variables. You can set them in your `.env` file (modify and rename `.env_sample` file to `.env`) or directly in your shell.
   ```
   GOOGLE_CLOUD_LOCATION=YOUR_CLOUD_LOCATION_NAME_HERE
   GOOGLE_CLOUD_PROJECT=YOUR_ROJECT_NAME_HERE
   GOOGLE_APPLICATION_CREDENTIALS=YOUR_GOOGLE_CREDENTIALS_FILEPATH_HERE
   ```

   You can find further configuration parameters in `src/config.py`. This incudes parameters such as prompt, max search length and llm model used by the service.

2. **Google Service Credentials**  
   Place your Google Cloud service account credentials JSON file in the `creds/` directory.

---

## Running the Service

```bash
uvicorn --reload src.main:app
```

---

## API Documentation

- Access the interactive API docs at:  
  [http://localhost:8000/docs](http://localhost:8000/docs)

---

## API Reference

### Search Endpoint

| Method | Endpoint         | Description                                      |
|--------|------------------|--------------------------------------------------|
| POST   | `/nlp/search`    | Extracts keywords from a user query.             |

**Request**

- **Headers:**
  - `Accept: application/json`
  - `Content-Type: application/json`
- **Body Parameters:**

  | Name     | Type    | Required | Description                                    |
  |----------|---------|----------|------------------------------------------------|
  | query    | string  | Yes      | The input text to extract keywords from.       |
  | synonyms | boolean | No       | Whether to include synonyms (default: false).  |

**Example Request**

```bash
curl -X POST 'http://localhost:8000/nlp/search' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "Tell me courses on noting and drafting",
    "synonyms": false
  }'
```

**Example Response**

```json
{
  "data": {
    "keywords": [
      {
        "keyword": "noting and drafting",
        "priority": 1
      }
    ]
  }
}
```

---

## Running with Docker

To run the application using Docker, follow these steps:

1. **Build the Docker image:**
    ```bash
    docker build -t nlp_search .
    ```

2. **Run the Docker container:**
    ```bash
    docker run -d -p 8000:8000 --name nlp_search_api nlp_search:latest
    ```

3. **Access the Application:**  
   Open your browser and navigate to [http://localhost:8000/docs](http://localhost:8000/docs) to access the FastAPI interactive documentation.

