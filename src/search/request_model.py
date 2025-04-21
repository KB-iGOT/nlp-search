from pydantic import BaseModel, Field, field_validator

class SearchModel(BaseModel):
    query: str
    synonyms: bool = Field(default=False)

    @field_validator('query')
    @classmethod
    def query_must_not_be_empty_and_length_constrained(cls, value):
        if not value:
            raise ValueError('Query cannot be empty.')
        if len(value) > 400:
            raise ValueError('Query cannot be longer than 400 characters.')
        return value
    