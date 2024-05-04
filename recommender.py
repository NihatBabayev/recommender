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

class Recommender:
    def __init__(self):
        """
        Initializes the Recommender class
        
        This method sets up the Spotify API credentials, loads and preprocesses the dataset, and creates the feature matrix & vector norms.
        """
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
        self.df["genre_encoded"] = self.label_encoder.fit_transform(self.df["genre"])  #encoding the genre column to get rid of the string values
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
        ] #these features were chosen by testing the impact of each one
        
        self.df[self.features] = self.scaler.fit_transform(self.df[self.features])

        # Creating and storing the feature matrix and vector norms  
        self.df["song_vector"] = self.df[self.features].values.tolist() #we store these values in the dataframe for later use
        self.song_matrix = self.df[self.features].values #this matrix is used to calculate cosine similarity
        self.song_norms = np.linalg.norm(self.song_matrix, axis=1) #this vector is used to calculate cosine similarity
        self.TOP = 20


    def classify_language(self, text):
        """
        This method determines the language using the characters used in song name.

        Args:
            text (_type_): Song name

        Returns:
            int: Corresponding language category
        """
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

    def unknown_song_vector(self, id):
        """
        This method retrieves the audio features of a song which's ID is not found in the dataset.
        
        Args:
            id (str): Spotify ID of the song
            
        Returns:
            np.array: Audio features of the song
        """
        #fetching the audio features of the song
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
        except Exception as e: #this exception may be useful if the song in playlist is not available in spotify's database, i.e. a local file
            print('EXCEPTION: Could not retrieve audio features for this song', e)
            return 

        genres = self.sp.artist(self.sp.track(id)['artists'][0]['id'])['genres'] #fetching the genre of the song

        if len(genres) == 0:
            genre_encoded = 72 #if no genre is found, we use the default genre 'pop'
        else:
            genre_candidates = []
            for genre in genres:
                try:
                    genre_encoded = self.label_encoder.transform([genre])[0] #encoding the genres 
                    genre_candidates.append(genre_encoded)
                except ValueError:
                    pass
            if genre_candidates:
                genre_encoded = np.mean(genre_candidates)
            else:
                genre_encoded = 72 #if the genre is not found, we use the default genre 'pop'

        language = self.classify_language(self.sp.track(id)['name'])
        audio_data.append(genre_encoded)
        audio_data.append(language)
        audio_data = np.array(audio_data).reshape(1, -1)
        audio_data = self.scaler.transform(audio_data) #scaling the audio features 

        return audio_data[0]  

    def unknown_song_matrix(self, candidate_unknown_ids):
        """
        This method uses multithreading to retrieve the audio features of multiple songs whichs' IDs are not found in the dataset.
        Then it creates a matrix of these audio features.
        
        Args:
            candidate_unknown_ids (list): List of Spotify IDs of the songs not found in the dataset
            
        Returns:
            np.array: Matrix of audio features of unknown songs
        """
        unknown_song_matrix = [] #this list will store the audio features of the songs which were not found in the database
        #we use ThreadPoolExecutor to fetch the audio features of the songs concurrently
        with ThreadPoolExecutor(max_workers=50) as executor: 
            future_to_track = {executor.submit(self.unknown_song_vector, i): i for i in candidate_unknown_ids}
            for future in as_completed(future_to_track):
                data = future.result()
                if data is not None:
                    unknown_song_matrix.append(data)

        return np.array(unknown_song_matrix)      

    def get_song_recommendations(self, song_query):
        """
        This method uses cosine similarity to find the most similar songs to the input song and returns the top 20 recommendations.
        Then it organizes the recommendations in a JSON format.
        
        Args:
            song_query (str): Song query to search on Spotify
            
        Returns: 
            dict: JSON format of the top 20 song recommendations
        """
        try:
            results = self.sp.search(q=song_query, limit=1)
            id = results['tracks']['items'][0]['id']
            song_name = results['tracks']['items'][0]['name']
            artist_name = results['tracks']['items'][0]['artists'][0]['name']
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
    
    def get_playlist_info(self, playlist_data, playlist_URI):
        """
        This method retrieves the info of the Spotify playlist.

        Args:
            playlist_data (dict): Spotify playlist data 
            playlist_URI (str): Spotify playlist URI

        Returns:
            tuple: User profile picture, username, playlist name, number of tracks, playlist cover image, duration
        """
        user_id = playlist_data["owner"]["id"]
        user_profile = self.sp.user(user_id)
        
        user_profile_picture = user_profile["images"][0]["url"]
        username = playlist_data["owner"]["display_name"]
        playlist_name = playlist_data["name"]
        number_of_tracks = playlist_data["tracks"]["total"]
        playlist_cover_image = playlist_data["images"][0]["url"]

        playlist_tracks = self.sp.playlist_tracks(playlist_URI)
        total_tracks = playlist_tracks["total"]
        tracks = playlist_tracks["items"]

        #the purpose of this while loop is to fetch all the tracks in the playlist, since the API only fetches 100 tracks at a time
        #it keeps fethcing until all the tracks are fetched, offset is used to keep track of the number of tracks fetched
        while len(tracks) < total_tracks:
            playlist_tracks = self.sp.playlist_tracks(playlist_URI, offset=len(tracks))
            tracks.extend(playlist_tracks["items"])
            
        #since the duration is returned in milliseconds, we convert it to hours and minutes
        duration = sum([track["track"]["duration_ms"] for track in tracks])
        hours, minutes = duration//3600000, duration % 3600000 // 60000
        duration = f"{hours} hours {minutes} minutes" if hours else f"{minutes} minutes"

        return user_profile_picture, username, playlist_name, number_of_tracks, playlist_cover_image, duration
        
    def get_playlist_recommendations_v1(self, playlist_url):
        """
        This method recommends songs for a Spotify playlist by averaging the audio features of the songs in the playlist.

        Args:
            playlist_url (str): Spotify playlist URL 

        Returns:
            dict: JSON format of the top 20 song recommendations
        """
        playlist_URI = playlist_url.split("/")[-1].split("?")[0]
        
        try:
            playlist_data = self.sp.playlist(playlist_URI)
        except Exception as e:
            return f"EXCEPTION: INVALID SPOTIFY PLAYLIST URL. {e}"
        
        user_profile_picture, username, playlist_name, number_of_tracks, playlist_cover_image, duration = self.get_playlist_info(playlist_data, playlist_URI)
        track_uris = [track["track"]["uri"] for track in playlist_data["tracks"]["items"]]
        ids_playlist = [uri.split(":")[-1] for uri in track_uris]

        candidate_known_ids = [] #this will store the audio features of the songs which were found in the database
        candidate_unknown_ids = [] #this list will store the track ids of the songs which were not found in local database

        for id in ids_playlist:
            try:
                input_song_vector_d = self.df[self.df['track_id'] == id]['song_vector'].values[0]
                candidate_known_ids.append(input_song_vector_d)
            except IndexError:
                candidate_unknown_ids.append(id)

        candidate_known_ids = np.array(candidate_known_ids)
        song_vectors = np.concatenate((candidate_known_ids, self.unknown_song_matrix(candidate_unknown_ids)), axis=0) #concatenating the known and unknown song vectors
        song_vectors = np.array([x for x in song_vectors if x is not None]) #removing the None values from the song vectors
        input_song_vector = np.mean(song_vectors, axis=0) #taking the average of the audio features of the songs in the playlist

        similarities = np.dot(self.song_matrix, input_song_vector) / (self.song_norms * np.linalg.norm(input_song_vector))#calculating the cosine similarity
        sorted_indices = np.argsort(similarities)[::-1][:self.TOP] #sorting the indices based on the similarity values
        top_songs = self.df.loc[sorted_indices, ['track_name', 'artist_name', 'track_id']] #fetching the top songs based on the similarity values
        top_songs['similarity'] = similarities[sorted_indices] #adding the similarity values to df
        top_songs = top_songs[~top_songs['track_id'].isin(ids_playlist)] #removing the songs which are already in the playlist

        #the code below is used to fetch the mp3, spotify and image urls of the top songs
        mp3_urls = []
        spotify_urls = []
        image_urls = []
        ids = top_songs['track_id'].tolist()
        results = self.sp.tracks(ids)

        for track in results['tracks']:
            mp3_urls.append(track['preview_url'])
            spotify_urls.append(track['external_urls']['spotify'])
            image_urls.append(track['album']['images'][0]['url'])

        top_songs['mp3_url'] = mp3_urls #not every song has a preview url provided by Spotify, so some of the values will be None
        top_songs['spotify_url'] = spotify_urls
        top_songs['image_url'] = image_urls

        final_json = {
            "username": username,
            "profile_picture": user_profile_picture,
            "playlist": playlist_name,
            "n_tracks": number_of_tracks,
            "image": playlist_cover_image,
            "duration": duration,
            "songs": json.loads(top_songs.head(self.TOP).to_json(orient="records"))
        }

        return final_json
    
    def get_playlist_recommendations_v2(self, playlist_url):
        """
        This method recommends songs for a Spotify playlist by finding the most similar songs to the songs in the playlist.

        Args:
            playlist_url (str): Spotify playlist URL 

        Returns:
            dict: JSON format of the top 20 song recommendations
        """
        playlist_URI = playlist_url.split("/")[-1].split("?")[0]
        
        try:
            playlist_data = self.sp.playlist(playlist_URI)
        except Exception as e:
            return f"EXCEPTION: INVALID SPOTIFY PLAYLIST URL. {e}"
        
        user_profile_picture, username, playlist_name, number_of_tracks, playlist_cover_image, duration = self.get_playlist_info(playlist_data, playlist_URI)
        track_uris = [track["track"]["uri"] for track in playlist_data["tracks"]["items"]]
        ids_playlist = [uri.split(":")[-1] for uri in track_uris]

        candidate_known_ids = [] #this will store the audio features of the songs which were found in the database
        candidate_unknown_ids = [] #this list will store the track ids of the songs which were not found in local database

        for id in ids_playlist:
            try:
                input_song_vector_d = self.df[self.df['track_id'] == id]['song_vector'].values[0]
                candidate_known_ids.append(input_song_vector_d)
            except IndexError:
                candidate_unknown_ids.append(id)

        candidate_known_ids = np.array(candidate_known_ids)
        song_vectors = np.concatenate((candidate_known_ids, self.unknown_song_matrix(candidate_unknown_ids)), axis=0)
        song_vectors = np.array([x for x in song_vectors if x is not None])
        song_vectors_transpose = song_vectors.T #transposing the song vectors (matrix)
        #calculating the cosine similarity between the songs in the database and all the songs in the playlist
        similarities_matrix = np.dot(self.song_matrix, song_vectors_transpose) / (self.song_norms[:, np.newaxis] * np.linalg.norm(song_vectors_transpose, axis=0)) 
        similarities_matrix_transpose = similarities_matrix.T
        
        c = {}
        
        #this for loop is used to find the top 5 songs for each song in the playlist
        #it will add the top 5 songs to the dictionary c, with the key being the index of the song in the playlist and the value being the similarity value
        for vector in similarities_matrix_transpose:
            indices = np.argsort(vector)[::-1][:5]
            values = vector[indices]
            for j in range(len(indices)):
                c[indices[j]] = values[j]
                
        sorted_c = dict(sorted(c.items(), key=lambda item: item[1], reverse=True))
        
        top_songs_new = pd.DataFrame()
        indices = []
        similarities = []
        
        #this loop is used to find the top 20 songs which are not in the playlist
        for key, value in sorted_c.items():
            if value < 1:
                indices.append(key)
                similarities.append(value)
            if len(indices) == 20:
                break
            
        #the code below is used to fetch the top song data using the indices 
        top_songs_new = self.df.loc[indices, ['track_name', 'artist_name', 'track_id']] 
        top_songs_new['similarity'] = similarities
        top_songs_new = top_songs_new[~top_songs_new['track_id'].isin(ids_playlist)]

        mp3_urls = []
        spotify_urls = []
        image_urls = []
        ids = top_songs_new['track_id'].tolist()
        results = self.sp.tracks(ids)

        for track in results['tracks']:
            mp3_urls.append(track['preview_url'])
            spotify_urls.append(track['external_urls']['spotify'])
            image_urls.append(track['album']['images'][0]['url'])

        top_songs_new['mp3_url'] = mp3_urls
        top_songs_new['spotify_url'] = spotify_urls
        top_songs_new['image_url'] = image_urls
        
        final_json = {
            "username": username,
            "profile_picture": user_profile_picture,
            "playlist": playlist_name,
            "n_tracks": number_of_tracks,
            "image": playlist_cover_image,
            "duration": duration,
            "songs": json.loads(top_songs_new.head(self.TOP).to_json(orient="records"))
        }

        return final_json
        
# def main():
#     recommender = Recommender()
#     print(recommender.get_playlist_recommendations_v2("https://open.spotify.com/playlist/73XPRn8DExoUaCGdQEWogX?si=1ba7dec3ade945f2"))
    
# if __name__ == "__main__":
#     main()
        
    
