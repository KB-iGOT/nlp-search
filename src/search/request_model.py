from pydantic import BaseModel, validator

class SearchModel(BaseModel):
    query: str
    synonyms: bool 

    @validator('query')
    def validate_query(cls, v):
        if len(v) > 500: # Explicit safety limit, though config has max_search_len
            raise ValueError('Query too long')
        # Basic check for specialized injection attempts (e.g. delimiters we might use)
        # Though we removed [] delimiters, keeping input clean is good practice.
        if "{{" in v or "}}" in v:
             raise ValueError('Query contains invalid characters')
        return v 
    