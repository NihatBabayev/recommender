import pandas as pd
import numpy as np
import json
import spotipy
import warnings
import os
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from spotipy.oauth2 import SpotifyClientCredentials
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')
load_dotenv()

class SongRecommender:
    def __init__(self):
        # Spotify API credentials
        self.cid = os.getenv("SPOTIFY_CLIENT_ID")
        self.secret = os.getenv("SPOTIFY_CLIENT_SECRET")
        self.client_credentials_manager = SpotifyClientCredentials(
            client_id=self.cid, client_secret=self.secret
        )
        self.sp = spotipy.Spotify(client_credentials_manager=self.client_credentials_manager)

        # Data loading and preprocessing
        self.df = pd.read_csv('final_data.csv')
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        self.df["genre_encoded"] = self.label_encoder.fit_transform(self.df["genre"])
        self.features = [
            "danceability",
            "energy",
            "loudness",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "tempo",
            "time_signature",
            "popularity",
            "genre_encoded",
            "language"
        ]
        self.df[self.features] = self.scaler.fit_transform(self.df[self.features])

        # Creating the feature matrix 
        self.df["song_vector"] = self.df[self.features].values.tolist()
        self.song_matrix = self.df[self.features].values
        self.song_norms = np.linalg.norm(self.song_matrix, axis=1)
        self.TOP = 20

    # Simple function to classify the language of a by its characters
    def classify_language(self, text):
        char_categories = {
            'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣ': 0,
            'あいうえおかきくけこ': 1,
            'ñ': 2,
            'éèàùìò': 3,
            'əğ': 4,
            'абвгдежзийклмнопрстуфхцчшщъыьэюя': 5
        }

        text_lower = text.lower()
        for chars, category in char_categories.items():
            if any(char in text_lower for char in chars):
                return category
        return 6

    # Function to get the audio features of a song if it is not in the dataset
    def unknown_song_vector(self, id):
        try : 
            results = self.sp.audio_features(id)
            audio_data = [
                results[0]['danceability'],
                results[0]['energy'],
                results[0]["loudness"],
                results[0]['speechiness'],
                results[0]['acousticness'],
                results[0]['instrumentalness'],
                results[0]['liveness'],
                results[0]['valence'],
                results[0]['tempo'],
                results[0]['time_signature'],
                self.sp.track(id)['popularity']
            ]
        except Exception as e: 
            print('EXCEPTION: Could not retrieve audio features for this song', e)
            return 

        genres = self.sp.artist(self.sp.track(id)['artists'][0]['id'])['genres']

        if len(genres) == 0:
            genre_encoded = 72
        else:
            genre_candidates = []
            for genre in genres:
                try:
                    genre_encoded = self.label_encoder.transform([genre])[0]
                    genre_candidates.append(genre_encoded)
                except ValueError:
                    pass
            if genre_candidates:
                genre_encoded = np.mean(genre_candidates)
            else:
                genre_encoded = 72

        language = self.classify_language(self.sp.track(id)['name'])

        audio_data.append(genre_encoded)
        audio_data.append(language)
        audio_data = np.array(audio_data).reshape(1, -1)
        audio_data = self.scaler.transform(audio_data)

        return audio_data[0]  

    def unknown_song_matrix(self, candidate_unknown_ids):
        unknown_song_matrix = []
        with ThreadPoolExecutor(max_workers=50) as executor:
            future_to_track = {executor.submit(self.unknown_song_vector, i): i for i in candidate_unknown_ids}
            for future in as_completed(future_to_track):
                data = future.result()
                if data is not None:
                    unknown_song_matrix.append(data)

        return np.array(unknown_song_matrix)      

    def get_song_recommendations(self, song_query):
        try:
            results = self.sp.search(q=song_query, limit=1)
            id = results['tracks']['items'][0]['id']
            song_name = results['tracks']['items'][0]['name']
            artist_name = results['tracks']['items'][0]['artists'][0]['name']
            # print(song_name, artist_name)
        except IndexError as e:
            return f"EXCEPTION: SONG NOT FOUND ON SPOTFIY. {e}"

        try:
            input_song_vector = np.array(self.df[self.df['track_id'] == id]['song_vector'].values[0])
        except IndexError:
            input_song_vector = self.unknown_song_vector(id)

        similarities = np.dot(self.song_matrix, input_song_vector) / (self.song_norms * np.linalg.norm(input_song_vector))
        sorted_indices = np.argsort(similarities)[::-1][:self.TOP]
        top_songs = self.df.loc[sorted_indices, ['track_name', 'artist_name', 'track_id']]
        top_songs['similarity'] = similarities[sorted_indices]

        mp3_urls = []
        spotify_urls = []
        image_urls = []
        ids = top_songs['track_id'].tolist()
        results = self.sp.tracks(ids)

        for track in results['tracks']:
            mp3_urls.append(track['preview_url'])
            spotify_urls.append(track['external_urls']['spotify'])
            image_urls.append(track['album']['images'][0]['url'])

        top_songs['mp3_url'] = mp3_urls
        top_songs['spotify_url'] = spotify_urls
        top_songs['image_url'] = image_urls

        if id not in ids:
            new_song = pd.DataFrame([{
                'track_name': self.sp.track(id)['name'],
                'artist_name': self.sp.track(id)['artists'][0]['name'],
                'track_id': id,
                'similarity': 1,
                'mp3_url': self.sp.track(id)['preview_url'],
                'spotify_url': self.sp.track(id)['external_urls']['spotify'],
                'image_url': self.sp.track(id)['album']['images'][0]['url']
            }])
            top_songs = pd.concat([new_song, top_songs], ignore_index=True)

        return json.loads(top_songs.head(self.TOP).to_json(orient="records"))
    
    def get_playlist_recommendations(self, playlist_url):
        playlist_link = playlist_url        
        playlist_URI = playlist_link.split("/")[-1].split("?")[0]
        
        try:
            print('hello')
            playlist_data = self.sp.playlist(playlist_URI)
            print('hello')
        except Exception as e:
            return f"EXCEPTION: INVALID SPOTIFY PLAYLIST URL. {e}"
        
        user_id = playlist_data["owner"]["display_name"]
        playlist_name = playlist_data["name"]
        number_of_tracks = playlist_data["tracks"]["total"]
        playlist_cover_image = playlist_data["images"][0]["url"]
        duration = sum([x["track"]["duration_ms"] for x in self.sp.playlist_tracks(playlist_URI)["items"]])
        hours, minutes = duration//3600000, duration % 3600000 // 60000
        duration = f"{hours} hours {minutes} minutes" if hours else f"{minutes} minutes" 
        

        # print(f"Username: {user_id}")
        # print(f"Playlist Name: {playlist_name}")
        # print(f"Number of Tracks: {number_of_tracks}")
        # print(f"Playlist Cover Image: {playlist_cover_image}")
        # print(f"Duration: {duration}")

        track_uris = [track["track"]["uri"] for track in playlist_data["tracks"]["items"]]
        ids_playlist = [uri.split(":")[-1] for uri in track_uris]

        candidate_known_ids = []
        candidate_unknown_ids = []

        for id in ids_playlist:
            try:
                input_song_vector_d = self.df[self.df['track_id'] == id]['song_vector'].values[0]
                candidate_known_ids.append(input_song_vector_d)
            except IndexError:
                candidate_unknown_ids.append(id)

        candidate_known_ids = np.array(candidate_known_ids)
        song_vectors = np.concatenate((candidate_known_ids, self.unknown_song_matrix(candidate_unknown_ids)), axis=0)
        song_vectors = np.array([x for x in song_vectors if x is not None])
        input_song_vector = np.mean(song_vectors, axis=0)

        similarities = np.dot(self.song_matrix, input_song_vector) / (self.song_norms * np.linalg.norm(input_song_vector))
        sorted_indices = np.argsort(similarities)[::-1][:self.TOP]
        top_songs = self.df.loc[sorted_indices, ['track_name', 'artist_name', 'track_id']] 
        top_songs['similarity'] = similarities[sorted_indices]
        top_songs = top_songs[~top_songs['track_id'].isin(ids_playlist)]

        mp3_urls = []
        spotify_urls = []
        image_urls = []
        ids = top_songs['track_id'].tolist()
        results = self.sp.tracks(ids)

        for track in results['tracks']:
            mp3_urls.append(track['preview_url'])
            spotify_urls.append(track['external_urls']['spotify'])
            image_urls.append(track['album']['images'][0]['url'])

        top_songs['mp3_url'] = mp3_urls
        top_songs['spotify_url'] = spotify_urls
        top_songs['image_url'] = image_urls

        final_json = {
            "username": user_id,
            "playlist": playlist_name,
            "n_tracks": number_of_tracks,
            "image": playlist_cover_image,
            "duration": duration,
            "songs": json.loads(top_songs.head(self.TOP).to_json(orient="records"))
        }

        return final_json
    
def main():

    recommender = SongRecommender()
    print(recommender.get_song_recommendations("the weeknd after horus"))
    
if __name__ == "__main__":
    main()
        
    