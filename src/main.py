from contextlib import asynccontextmanager
from fastapi import FastAPI
from .services.redis_service import redis_service
from .config import get_settings
from .search.router import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis_service.init_cache(get_settings())
    yield
    redis_service.close_cache()


app = FastAPI(lifespan=lifespan)

app.include_router(router, prefix="/nlp")

@app.get("/")
def welcome():
    return {"message" : "Welcome to NLP Search Service!"}
