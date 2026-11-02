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

3. **LLM timeouts, retries and fail fast**

   The model is called on Vertex AI through the `google-genai` SDK. The SDK's own retries are
   left off, so it makes one attempt per call; all retrying is done by the service itself, with
   `tenacity`.

   **Fail fast.** A request whose answer is not cached waits at most `RESPONSE_TIMEOUT_SECONDS`
   for the model. If the answer is not ready by then, the client gets a `202` asking it to try
   again (see [Responses](#responses)), while the model call carries on in the background and
   caches its answer in Redis. The client's next try is then served from the cache. A request
   for a query that is already being fetched waits on that same call rather than calling the
   model again. This is per worker process: with several workers or pods, each may make its own
   call for the same query. On shutdown, the service waits up to 20 seconds for calls still
   running so that their answers are cached rather than lost.

   Each attempt is limited to `LLM_ATTEMPT_TIMEOUT_SECONDS`. An attempt still running at the limit
   is cancelled and retried like any other timeout. Rate limits (`429`), server errors (`500`,
   `502`, `503`, `504`), timeouts (including a `408` from Vertex AI) and dropped connections
   are retried, as is a response that
   arrives but is not readable JSON. Errors the call cannot recover from (bad credentials, an
   invalid request, a response stopped by the safety filters) fail immediately.

   Once the retries are used up, the call fails with the status below. The client only receives
   it if the call ends within its `RESPONSE_TIMEOUT_SECONDS`; otherwise it has already had its
   `202`, the failure is logged, and nothing is cached, so the next request tries again.

   | Final failure | Status |
   |---------------|--------|
   | Vertex AI returned `429`, `503` or `504` | the same status |
   | Vertex AI returned any other error | `502` |
   | The attempt timed out, or Vertex AI returned `408` | `504` |
   | Vertex AI could not be reached | `503` |
   | The response was empty (e.g. blocked) or not readable JSON | `502` |

   > Each retry is a fresh billed call against your regional quota. Because `TEMPERATURE=0`
   > makes the same prompt return the same text, retrying an unreadable response often spends
   > every attempt to get the same answer. Watch for the `Unusable LLM response` log line.

   | Variable | Default | Description |
   |----------|---------|-------------|
   | `RESPONSE_TIMEOUT_SECONDS` | `3` | Seconds a request waits for an answer that is not cached before returning `202`. |
   | `LLM_MAX_RETRIES` | `2` | Retries after the first call, so `2` means up to 3 calls. `0` disables retrying. |
   | `LLM_ATTEMPT_TIMEOUT_SECONDS` | `10` | Seconds allowed for one call to the model. Each retry gets the same again. |

   The wait between attempts starts at 1 second and doubles each time, up to 4 seconds, with up
   to 1 second of random jitter added so that callers rate limited together do not retry
   together. With the defaults, a fully failing call takes at most about
   `3 × 10s + (1 + 2)s + up to 2s of jitter ≈ 35` seconds before giving up. No client waits that
   long, since that time is spent in the background. Keep `LLM_ATTEMPT_TIMEOUT_SECONDS` well above
   how long a slow answer takes, or slow queries are cut off on every attempt and never cached.

4. **Redis response cache (optional)**

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

   Caching adds no endpoints, request fields or response headers. Its one visible effect is the `202` described
   above, which tells the client the answer is being cached for its next try. Cache hits,
   misses and Redis problems appear in the service logs, and the stored records can be read
   directly from Redis.

5. **What a record looks like**

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

#### Responses

Only `200` carries keywords. Every other status has a body of the form `{"detail": "..."}`.

| Status | When | What the client should do |
|--------|------|---------------------------|
| `200` | Keywords found, from the cache or the model. | Use `data`. |
| `202` | The answer was not ready within `RESPONSE_TIMEOUT_SECONDS` and is still being fetched. | Show `detail`, or retry after a couple of seconds; the answer will usually be cached by then. |
| `400` | The query is empty or only whitespace. | Fix the request. |
| `422` | The body is invalid: `query` missing, or longer than 400 characters. | Fix the request. |
| `429` | Vertex AI rate limit or quota, after all retries. | Retry later. |
| `502` | The model failed, or its answer was empty or unreadable. | Retry later. |
| `503` | Vertex AI could not be reached, or the service is shutting down. | Retry later. |
| `504` | Every attempt timed out, or Vertex AI itself returned `408` or `504`. | Retry later. |
| `500` | Unexpected error in the service. | Retry later; check the service logs. |

**Example `202` Response**

```http
HTTP/1.1 202 Accepted

{"detail": "This is taking longer than expected. Please try again."}
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

