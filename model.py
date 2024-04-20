import pandas as pd
import numpy as np
import spotipy
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from spotipy.oauth2 import SpotifyClientCredentials
warnings.filterwarnings('ignore')

class SongRecommender:
    def __init__(self):
        # Spotify API credentials
        self.cid = "d9219a18c2ed48e685ea287cbfcdda95"
        self.secret = "b9492ce5ac57427c9ee60be103779a29"
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

        def classify_language(text):
            if any(char in text.lower() for char in ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']):
                return 0
            elif any(char in text.lower() for char in ['あ', 'い', 'う', 'え', 'お', 'か', 'き', 'く', 'け', 'こ']):
                return 1
            elif 'ñ' in text:
                return 2
            elif any(char in text.lower() for char in ['é', 'è', 'à', 'ù', 'ì', 'ò']):
                return 3
            elif any(char in text.lower() for char in ['ə', 'ğ']):
                return 4
            elif any(char in text.lower() for char in [
                'а', 'б', 'в', 'г', 'д', 'е', 'ё', 'ж', 'з', 'и', 'й', 'к', 'л', 
                'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 
                'ъ', 'ы', 'ь', 'э', 'ю', 'я']):
                return 5
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

    def get_top_similar_songs(self, search_string):
        try:
            results = self.sp.search(q=search_string, limit=1)
            id = results['tracks']['items'][0]['id']
            song_name = results['tracks']['items'][0]['name']
            artist_name = results['tracks']['items'][0]['artists'][0]['name']
            print(song_name, artist_name)
        except IndexError:
            print("EXCEPTION: Song not found on Spotify.")
            return

        try:
            input_song_vector = np.array(self.df[self.df['track_id'] == id]['song_vector'].values[0])
        except IndexError:
            input_song_vector = self.unknown_song_vector(id)

        TOP = 20
        similarities = np.dot(self.song_matrix, input_song_vector) / (self.song_norms * np.linalg.norm(input_song_vector))
        sorted_indices = np.argsort(similarities)[::-1][:TOP]
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

        return top_songs.head(TOP).to_json(orient="records")
        
