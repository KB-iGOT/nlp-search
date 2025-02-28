from pydantic import BaseModel

class SearchModel(BaseModel):
    query: str
    synonyms: bool 
    