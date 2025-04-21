from fastapi import FastAPI 
from .search.router import router
app = FastAPI()

app.include_router(router, prefix="/nlp")

@app.get("/")
def welcome():
    return {"message" : "Welcome to NLP Search Service!"}

