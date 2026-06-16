import os, json
import logging
from fastapi.exceptions import HTTPException 
import vertexai
from vertexai.generative_models import GenerativeModel
from functools import lru_cache
from src import config
from src.search.request_model import SearchModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@lru_cache
def get_settings():
    return config.Settings()

settings = get_settings()
if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=settings.GOOGLE_APPLICATION_CREDENTIALS

vertexai.init(project=settings.GOOGLE_CLOUD_PROJECT, location=settings.GOOGLE_CLOUD_LOCATION)
model = GenerativeModel(
        settings.MODEL_NAME,
        system_instruction=[
            "You are a helpful language expert.",
            "Your mission is to extract search keywords from queries."
        ],
        generation_config= {
            "max_output_tokens": int(settings.MAX_OUTPUT_TOKENS),
            "temperature": float(settings.TEMPERATURE),
            "top_p": float(settings.TOP_P),
            "top_k": int(settings.TOP_K)
        }
) 

def search_request(req_data: SearchModel):
    try:
        logger.info(f"Received nlp serch request :: {req_data.model_dump()}")
        
        if not req_data.query.strip():
            return HTTPException(status_code=400, detail="Query cannot be empty.")
        
        if len(req_data.query) > int(settings.MAX_SEARCH_LEN):
            return HTTPException(status_code=400, detail=f"Query cannot be longer than {int(settings.MAX_SEARCH_LEN)} characters")
            
        response = llm_request(req_data)
        if isinstance(response, Exception):
            return response
        
        logger.info(f"Response :: {response}")
        return {"data" : response}
    except Exception:
        logger.exception("An unexpected error occurred")
        raise HTTPException(status_code=500, detail="Internal server error during request processing.")
    

def llm_request(req_data: SearchModel):
    
    instruction = settings.NLP_SEARCH_INSTRUCTION_PROMPT
    example = settings.NPL_SEARCH_EXAMPLE_PROMPT

    if req_data.synonyms:
        instruction = instruction.replace(']' , '] \n Add synonym for keywords wherever possible.')
        example = example.replace(']' , '] \n Add synonym for keywords wherever possible.')

    prompt = instruction + req_data.query + example
    
    logger.info(f"Final prompt :: {prompt}")

    responses = model.generate_content(
        prompt,
        stream=True
    )
    res_text_designation = ""
    for response in responses:
        res_text_designation += response.text
    
    logger.info(f"Model Response :: {res_text_designation}")
    
    try:
        return json.loads(res_text_designation.replace('```','').replace('json', ''))
    except json.JSONDecodeError:
        logger.exception("Failed to decode LLM response: %s", res_text_designation)
        return HTTPException(status_code=500, detail="Failed to process the query. Please try again later!")
    except Exception:
        logger.exception("An unexpected error occurred during LLM processing")
        logger.error("Raw LLM response: %s", res_text_designation)
        return HTTPException(status_code=500, detail="Failed to process the query. Please try again later!")