import traceback
from fastapi.exceptions import HTTPException
from src import config 
import logging
import vertexai
from vertexai.generative_models import GenerativeModel
import os, json, yaml
from functools import lru_cache

logging.basicConfig()
logger = logging.getLogger(__name__)


@lru_cache
def get_settings():
    return config.Settings()

@lru_cache
def get_prompts():
    with open("src/prompts.yaml", "r") as f:
        return yaml.safe_load(f)


settings = get_settings()
prompts = get_prompts()
if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=settings.GOOGLE_APPLICATION_CREDENTIALS

vertexai.init(project=settings.PROJECT, location=settings.LOCATION)
model = GenerativeModel(
        settings.MODEL,
        system_instruction=[
            "You are a helpful language expert.",
            "Your mission is to extract search keywords from queries.",
        ],
) 

generation_config = {
    "max_output_tokens": settings.MAX_OUTPUT_TOKENS,
    "temperature": settings.TEMPERATURE,
    "top_p": settings.TOP_P,
    "top_k": settings.TOP_K,
    "response_mime_type": "application/json",
    "response_schema": {
        "type": "OBJECT",
        "properties": {
            "keywords": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "keyword": {"type": "STRING"},
                        "priority": {"type": "INTEGER"},
                         "synonyms": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"}
                        }
                    },
                    "required": ["keyword", "priority"]
                }
            }
        }
    }
}


def search_request(req_data):
    try:
        logger.info(req_data)
        query = req_data.query
        if query.strip() == '' or len(query) > settings.MAX_SEARCH_LEN:
            return HTTPException(status_code=400, detail="Empty query string or query too long. Current max limit " + str(settings.MAX_SEARCH_LEN))
        synonym = False
        if req_data.synonyms:
            synonym = req_data.synonyms
        logger.info(query)
        response = llm_request(query, synonym)
        if isinstance(response, Exception):
            return response
        for keyword in response["keywords"]:
            logger.info(keyword)
        return {"data" : response}
    except Exception as e:
        traceback.print_exc()
        return HTTPException(status_code=400, detail="Error")
    

def llm_request(query, synonym):
    version = settings.PROMPT_VERSION
    if version not in prompts:
        if "latest" in prompts:
             version = prompts["latest"]
        else:
             raise ValueError(f"Prompt version {version} not found in prompts.yaml")
    
    # Handle alias if the version points to a string (another version)
    if isinstance(prompts.get(version), str):
        version = prompts[version]

    selected_prompt = prompts[version]
    prompt = f"{selected_prompt['instruction']} {query} {selected_prompt['example']}"
    
    logger.info(synonym)
    if synonym:
        prompt += "\n Instruction: Add synonym for keywords wherever possible."
    logger.info(prompt)
    responses = model.generate_content(
        prompt,
        generation_config=generation_config,
        #safety_settings=safety_settings,
        stream=True,
    )
    res_text_designation = ""
    for response in responses:
        res_text_designation += response.text
    logger.info(res_text_designation)

    try:
        return json.loads(res_text_designation)
    except Exception as e:
        logger.error(res_text_designation)
        traceback.print_exc()
        return HTTPException(status_code=500, detail="LLM response parsing issue")