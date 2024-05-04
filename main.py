from fastapi import FastAPI, APIRouter
from recommender import Recommender

app = FastAPI()
recommender = Recommender()
recommendations_router = APIRouter()

@recommendations_router.post("/songs/")
async def get_song_recommendations(song_query: str):
    return recommender.get_song_recommendations(song_query)

@recommendations_router.post("/playlists_v1/")
async def get_playlist_recommendations_v1(playlist_url: str):
    return recommender.get_playlist_recommendations_v1(playlist_url)

@recommendations_router.post("/playlists_v2/")
async def get_playlist_recommendations_v2(playlist_url: str):
    return recommender.get_playlist_recommendations_v2(playlist_url)

app.include_router(recommendations_router, prefix="/recommendations", tags=["recommendations"])
