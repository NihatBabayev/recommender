from fastapi import FastAPI
from model import SongRecommender
import json

app = FastAPI()
recommender = SongRecommender()

@app.get("/getsongrecommendations/")
async def get_song_recommendations(song_query: str):
    return recommender.get_song_recommendations(song_query)

@app.get("/getplaylistrecommendations/")
async def get_playlist_recommendations(playlist_url: str):
    return recommender.get_playlist_recommendations(playlist_url)
