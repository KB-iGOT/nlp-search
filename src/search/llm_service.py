import traceback
from fastapi.exceptions import HTTPException
from src import config 
import logging
import vertexai
from vertexai.generative_models import GenerativeModel
import os, json
from functools import lru_cache

logging.basicConfig()
logger = logging.getLogger(__name__)


@lru_cache
def get_settings():
    return config.Settings()

settings = get_settings()
if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=settings.GOOGLE_APPLICATION_CREDENTIALS

vertexai.init(project=settings.project, location=settings.location)
model = GenerativeModel(
        settings.model,
        system_instruction=[
            "You are a helpful language expert.",
            "Your mission is to extract search keywords from queries.",
        ],
) 

generation_config = {
    "max_output_tokens": settings.max_output_tokens,
    "temperature": settings.temperature,
    "top_p": settings.top_p,
    "top_k": settings.top_k,
}


def search_request(req_data):
    try:
        logger.info(req_data)
        query = req_data.query
        if query.strip() == '':
            return HTTPException(status_code=400, detail="Empty query string")
        synonym = False
        if req_data.synonyms:
            synonym = req_data.synonyms
        logger.info(query)
        response = llm_request(query, synonym)

        for keyword in response["keywords"]:
            logger.info(keyword)
        return {"data" : response}
    except Exception as e:
        traceback.print_exc()
        return HTTPException(status_code=400, detail="Error")
    

def llm_request(query, synonym):
    
    prompt = settings.nlp_search_instruction_prompt + query + settings.nlp_search_example_prompt
    logger.info(synonym)
    if synonym:
        prompt = prompt.replace(']' , '] \n Add synonym for keywords wherever possible.')
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
    return json.loads(res_text_designation.replace('```','').replace('json', ''))