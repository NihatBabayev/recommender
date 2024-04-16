from fastapi import FastAPI
from model import SongRecommender

app = FastAPI()
recommender = SongRecommender()

@app.post("/getrecommendation/")
async def hello_person(search_string: str):
    return recommender.get_top_similar_songs(search_string)
