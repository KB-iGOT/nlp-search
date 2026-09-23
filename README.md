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

   WEB_CONCURRENCY=Number of worker processes
   ```

   You can find further configuration parameters in `src/config.py`. This incudes parameters such as prompt, max search length and llm model used by the service.

2. **Google Service Credentials**  
   Place your Google Cloud service account credentials JSON file in the `creds/` directory.

3. **Redis response cache (optional)**

   Responses are deterministic (`TEMPERATURE=0`), so identical queries are served from Redis
   instead of calling the LLM again. The cache is **on by default**; set `REDIS_ENABLED=false`
   to turn it off.

   | Variable | Default | Description |
   |----------|---------|-------------|
   | `REDIS_ENABLED` | `true` | Master switch. When false the service never touches Redis. |
   | `REDIS_HOST` / `REDIS_PORT` / `REDIS_DB` | `localhost` / `6379` / `0` | Connection target. |
   | `REDIS_PASSWORD` / `REDIS_SSL` | empty / `false` | Leave unset for an unauthenticated in-cluster Redis. |
   | `REDIS_CACHE_TTL_DAYS` | `30` | How long a cached response lives. |
   | `REDIS_QUERY_COUNTER_ENABLED` | `true` | Count how often each query is asked. |
   | `REDIS_KEY_PREFIX` | `nlpsearch` | Prefix for every key written by this service. |

   > **Cached responses do not expire when the prompts or `MODEL_NAME` change.** Only the query
   > text and the `synonyms` flag form the cache key, so after editing either one, clear the
   > cache manually or previous answers keep being served for up to `REDIS_CACHE_TTL_DAYS`:
   >
   > ```bash
   > redis-cli --scan --pattern "nlpsearch:*" | xargs -r redis-cli del
   > ```

   The cache is best-effort: if Redis is unreachable the request falls through to the LLM and
   still succeeds. After a few consecutive failures the service stops calling Redis for a
   cooldown so an outage does not add connection timeouts to every request, then retries
   automatically.

   **Running Redis locally with Docker:**

   ```bash
   docker run -d --name nlp-redis -p 127.0.0.1:6379:6379 --restart unless-stopped \
     redis:6.2-alpine redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru
   ```

   Useful while developing:

   ```bash
   docker exec nlp-redis redis-cli --scan --pattern "nlpsearch:*"   # what is cached
   docker exec nlp-redis redis-cli flushdb                          # clear the cache
   ```

   Caching is invisible to callers: the API is unchanged, with no added endpoints, request
   fields or response headers. Cache hits, misses and Redis problems appear in the service
   logs, and the stored records can be read directly from Redis.

4. **What a record looks like**

   One Redis hash per query holds both the cached answer and how often it has been asked:

   ```bash
   $ redis-cli hgetall "nlpsearch:v1:nosyn:92b9035c..."
   count           3
   query           give me python courses
   data            {"keywords": [{"keyword": "python courses", "priority": 1}]}
   ```

   Queries are lowercased, trimmed and their internal whitespace collapsed before anything is
   keyed or stored, so `"  GIVE me   Python Courses "` and `"give me python courses"` are one
   record, and Redis holds only the cleaned form.

   `count` is incremented by Redis itself, so it stays correct with several workers or pods
   running, and it rises on cache hits too, not just on calls to the model. The whole record
   expires `REDIS_CACHE_TTL_DAYS` after the query was first asked, counter included, and can
   also be evicted early under `maxmemory-policy allkeys-lru`. Counts are a usage signal, not
   an audit record.

   Because `synonyms` is part of the key, the same question asked with and without synonyms is
   two records with separate counts.

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

