from fastapi import FastAPI
from model import SongRecommender
import json

app = FastAPI()
recommender = SongRecommender()

@app.post("/getrecommendation/")
async def hello_person(search_string: str):
    result = recommender.get_top_similar_songs(search_string)
    return json.loads(result)
