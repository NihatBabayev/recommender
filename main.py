from fastapi import FastAPI, APIRouter
from model import SongRecommender

app = FastAPI()
recommender = SongRecommender()
recommendations_router = APIRouter()

@recommendations_router.post("/songs/")
async def get_song_recommendations(song_query: str):
    return recommender.get_song_recommendations(song_query)

@recommendations_router.post("/playlists/")
async def get_playlist_recommendations(playlist_url: str):
    return recommender.get_playlist_recommendations(playlist_url)

app.include_router(recommendations_router, prefix="/recommendations", tags=["recommendations"])
