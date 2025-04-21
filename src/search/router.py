from src.search.request_model import SearchModel
from fastapi import APIRouter
from src.search.llm_service import search_request


router = APIRouter()

@router.post("/search")
def search(search_model : SearchModel):
    return search_request(search_model)