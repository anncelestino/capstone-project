# Imports
import pandas as pd
import urllib.request
import requests
import time
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff


# Imports for Classification Modelling
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay

# Extra Imports (GIFS and Music)
from pandas import json_normalize
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_lottie import st_lottie
from streamlit_player import st_player

# Spotipy Client
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
SPOTIPY_CLIENT_ID='b7e28926171e4082a9d75d0b0c53d103'
SPOTIPY_CLIENT_SECRET='58336c49f6e549318c9780a5c5067e43'
SPOTIPY_REDIRECT_URI='https://localhost:5000/callback'

auth_manager = SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)

#Imports from Folder
import polarplot
import songrecommendations

# Reading CSVs
original_df = pd.read_csv('data/song_data.csv')
df = pd.read_csv('data/cleaned_song_data.csv')



# Set page title and icon
st.set_page_config(page_title = "Song Analyzer App", page_icon = '🎙️', layout = 'wide')



# Sidebar
st.sidebar.image("https://images.ctfassets.net/23aumh6u8s0i/2Qhstbnq6i34wLoPoAjWoq/9f66f58a22870df0d72a3cbaf77ce5b6/streamlit_hero.jpg", width = 275, caption = 'Built with Streamlit 🎈')
st.sidebar.subheader("**:orange[Page Selection]**")
page = st.sidebar.selectbox("Select a page", ["Introduction 👋🏻", "Spotify API 🎧", "The Song Popularity Dataset 📑", "Explore the Dataset 📊", "Machine Learning Modeling 🤖", "Predict Song Popularity 🔮"])


definition_choices = ['Acousticness', 'Danceability', 'Energy', 'Instrumentalness', 'Key', 'Liveness', 'Loudness', 'Speechiness', 'Tempo', 'Time Signature', 'Valence', 'Song Duration']

def definition(definition_selected):
    if definition_selected == 'Acousticness':
        return f'A confidence measure from 0.0 to 1.0 of whether the track is acoustic. Acoustic means the music is created without the use of any electronic amplification or effects. It\'s just the pure, raw sound of the instruments. 1.0 represents high confidence the track is acoustic.'
    if definition_selected == 'Danceability':
        return f'Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.'
    if definition_selected == 'Energy':
        return f'Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.'
    if definition_selected == 'Instrumentalness':
        return f'Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.'
    if definition_selected == 'Key':
        return f'The key the track is in. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1.'
    if definition_selected == 'Liveness':
        return f'Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.'
    if definition_selected == 'Loudness':
        return f'The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typically range between -60 and 0 db.'
    if definition_selected == 'Speechiness':
        return f'Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.'
    if definition_selected == 'Tempo':
        return f'The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.'
    if definition_selected == 'Time Signature':
        return f'A sign used to indicate musical meter and usually written with one number above another with the bottom number indicating the kind of note used as a unit of time and the top number indicating the number of these units in each measure.'
    if definition_selected == 'Valence':
        return f'A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).'
    if definition_selected == 'Song Duration':
        return f'THe amount of time an entire piece of work lasts'
    if definition_selected == 'Audio Mode':
        return f'Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.'





# Build a homepage
if page == "Introduction 👋🏻":

    col1, col2, col3_spacer = st.columns([3, 1.5, 0.1])

    with col1:
        
        # container = st.container(border = True)
        # with container:
            st.title(" :violet[Explore and Analyze Your Songs] ♩♫♪♬")
            st.subheader(":grey[With] ✨ :rainbow[The Song Stats App] ✨")
            st.divider()
            st.markdown("👋🏻 Hello there! Welcome to Ann's Song Analysis Explorer App. Would you like to know if your next song will become a hit? With this app, you have the ability to predict whether or not your song will gain popularity by looking at specific musical features based off the [Song Popularity Dataset](https://www.kaggle.com/datasets/yasserh/song-popularity-dataset/data?select=song_data.csv).")  
            st.markdown("This multipurpose app also allows you to explore millions of songs from :green[Spotify] utilizing Spotify's API. You can search for specific songs, artists, or albums to visualize the different features of each song as well as preview songs if available. Recommendations of similar songs can be easily found with a click of a button!")
    
    with col2:

        container2 = st.container(border = True)
        with container2:
            st.subheader(":rainbow[A Streamlit web app] :grey[by] [Ann Celestino](https://www.linkedin.com/in/ann-daniel-celestino-459333184)")
            st.divider()
            st.write("###")
            st.write("##")
            # Loading Music Icon
            def load_lottieurl(url: str):
                r = requests.get(url)
                if r.status_code != 200:
                    return None
                return r.json()
        
            music_icon = load_lottieurl("https://lottie.host/f0ea1674-80b8-4ff2-b449-1707e96347d0/ppNEFOgf5t.json")
            st_lottie(music_icon, speed = 1, height = 200)
    st.divider()

    col1_spacer, col1, col2, col3_spacer = st.columns([0.1, 2, 1, 0.1])
    with col1:
        st.image("https://5.imimg.com/data5/SELLER/Default/2023/5/309879614/LG/WV/VC/72240390/audio-recording-studio-service.jpg")
    
    with col2:
        # container3 = st.container(border = True)
        st.markdown("**:red[Here you can:]**")
        st.markdown('- Search for songs, artists, and albums') 
        st.markdown('- Visualize different features of each song') 
        st.markdown('- Get recommendations of similar songs') 
        st.markdown('- Explore a Song Popularity Dataset')
        st.markdown('- Make Analyses')
        st.markdown('- Learn about different Machine Learning Models') 
        st.markdown('- Predict whether a song will gain popularity')
    st.divider()
    container = st.container(border=True)
    container.write("💡 **:red[TIP]**: Get started by opening the sidebar by clicking on the arrow on the top left corner. I recommend keeping the sidebar open as more options will pop-up as you explore the app. :orange[Enjoy]! 🎈")
    st.write("***This app was originally created for my Capstone Project at Coding Temple for the purpose of creating an interactive and user-friendly app with the primary focus of analyzing and using machine learning models to make predictions.***")
    container2 = st.container(border=True)

    
    # Use local CSS

    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)


    local_css("style/style.css")

    animation_symbol = "♪"
    animation_symbol2 = "♫"
    animation_symbol3 = "♬"
    animation_symbol4 = "♩"

    st.markdown(
        f"""
        <div class="snowflake">{animation_symbol}</div>
        <div class="snowflake">{animation_symbol4}</div>
        <div class="snowflake">{animation_symbol2}</div>
        <div class="snowflake">{animation_symbol3}</div>
        <div class="snowflake">{animation_symbol}</div>
        <div class="snowflake">{animation_symbol4}</div>
        <div class="snowflake">{animation_symbol2}</div>
        <div class="snowflake">{animation_symbol3}</div>
        """,
        unsafe_allow_html=True
    )











if page == 'Spotify API 🎧':

    col1, col1_spacer, col2, col2_spacer = st.columns((2, 0.2, 0.5, 0.2))
    with col1:
        st.title(":green[Spotify] :rainbow[Song Search]")
        st.write("In this page you can choose a song, album, or artist to view specific features for each. This page uses Spotify's API to generate the songs and data information. :green[Start] by picking a search choice in the sidebar or entering a keyword from a song down below.")
    with col2:
        st.image("https://play-lh.googleusercontent.com/cShys-AmJ93dB0SV8kE6Fl5eSaf4-qMMZdwEDKI5VEmKAXfzOqbiaeAsqqrEBCTdIEs")
    
    st.divider()
    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    
    music_icon = load_lottieurl("https://lottie.host/a97dff1e-c79e-41e7-8d53-9754bab61274/ILXLNH6kBg.json")
    st_lottie(music_icon, speed = 1, height = 75)

    st.sidebar.divider()
    st.sidebar.subheader(":green[Spotify] Search Options")
    search_choices = ['Song/Track 🎧', 'Artist 🎸', 'Album 💿']
    search_selected = st.sidebar.selectbox("Search choice ", search_choices)

    col01, col02, col03 = st.columns((8, 7, 5))
    with col01:
        st.subheader(f":green[{search_selected} Selection]")
    search_keyword = st.text_input(search_selected + " (Keyword Search)")
    button_clicked = st.button ("Search")


    # definition_selected = st.sidebar.selectbox("Select a feature ", definition_choices)


    # container1 = st.container(border = True)
    search_results = []
    tracks = []
    songs = []
    artists = [] 
    albums = []


    # with container1:
    if search_keyword is not None and len(str(search_keyword)) > 0:
        if search_selected == 'Song/Track 🎧':
            st.write("**:green[Search from list of songs]**")
            tracks = sp.search(q='track:'+search_keyword,type='track', limit=20)
            tracks_list = tracks['tracks']['items']
            if len(tracks_list) > 0:
                    for track in tracks_list:
                        # st.write(f"{track['name']}- By :blue[{track['artists'][0]['name']}]")
                        search_results.append(track['name'] + "- By -" + track['artists'][0]['name'])
        elif search_selected == 'Artist 🎸':
            st.divider()
            st.write("**:green[Search from list of artists]**")
            # container = st.container(border = True)
            # with container:
            artists = sp.search(q='artist:'+search_keyword,type='artist', limit=20)
            artists_list = artists['artists']['items']
            if len(artists_list) > 0:
                for artist in artists_list:
                    # st.write(f"{artist['name']}")
                    search_results.append(artist['name'])
        elif search_selected == 'Album 💿':
            st.divider()
            st.write("**:green[Search from list of albums]**")
            albums = sp.search(q='album:'+search_keyword,type='album', limit=20)
            albums_list = albums['albums']['items']
            if len(albums_list) > 0:
                for album in albums_list:
                    # st.write(f"{album['name']}- By :blue[{album['artists'][0]['name']}]")
                    # print("Album ID: " + album['id'] + " / Artist ID - " + album['artists'][0]['id'])
                    search_results.append(album['name'] + "- By -" + album['artists'][0]['name'])

        selected_album = None
        selected_artist = None
        selected_track = None
        if search_selected == 'Song/Track 🎧':
            selected_track = st.selectbox("Select a song/track", search_results)
        elif search_selected == 'Artist 🎸':
            selected_artist = st.selectbox("Select an artist ", search_results)
        elif search_selected == 'Album 💿':
            selected_album = st.selectbox("Select an album ", search_results)


        if selected_track is not None and len(tracks) > 0:
            tracks_list = tracks['tracks']['items']
            track_id = None
            track_uri = None
            if len(tracks_list) > 0:
                for track in tracks_list:
                    str_temp = track['name'] + "- By -" + track['artists'][0]['name']
                    if str_temp == selected_track:
                        track_id = track['id']
                        track_uri = track['uri']
                        # album_id = album['id']
                        # album_uri = album['uri']
                        # album_name = album['name']
                        track_album = track['album']['name']
                        img_album = track['album']['images'][1]['url']
                        songrecommendations.save_album_image(img_album, track_id)
                        song_preview = track['preview_url']
                            
            selected_track_choice = None
            if track_id is not None:
                # albums_list = albums['albums']['items']
                image = songrecommendations.get_album_image(track_id)
                st.image(image)
                if song_preview is not None:
                    st.audio(song_preview, format = 'audio/mp3')
                st.divider()
                track_choices = ['Song Features ♭', 'Similar Songs Recommendations 🩵']
                selected_track_choice = st.sidebar.selectbox('More options', track_choices)

                if selected_track_choice == 'Song Features ♭':
                    with st.container(border = True):
                        track_features = sp.audio_features(track_id)
                        df2 = pd.DataFrame(track_features, index = [0])
                        df2_features = df2.loc[: ,['acousticness', 'danceability', 'energy', 'instrumentalness',     'liveness', 'speechiness', 'valence']]
                        col1, col2, = st.columns((5,5))
                        col1.subheader(":green[Audio Features]")
                        col1.dataframe(df2_features)
                        col2.subheader(":green[Polarplot]")
                        with col2:
                            polarplot.feature_plot(df2_features)


                elif selected_track_choice == 'Similar Songs Recommendations 🩵':
                    with st.container(border = True):
                        token = songrecommendations.get_token(SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET)
                        similar_songs_json = songrecommendations.get_track_recommendations(track_id, token)
                        recommendation_list = similar_songs_json['tracks']
                        recommendation_list_df = pd.DataFrame(recommendation_list)
                        recommendation_df = recommendation_list_df[['name', 'explicit', 'duration_ms', 'popularity']]
                        st.subheader(":green[Similar Songs Recommendation]")
                        st.dataframe(recommendation_df)
                        st.subheader(":green[Similar Songs Graph]")
                        st.write("Red circle means explicit. Size of circle based on popularity.")
                        songrecommendations.song_recommendation_vis(recommendation_df)
            else:
                st.write("Please select a track from the list")
     
            tip_container = st.container(border = True)
            tip_container.write("💡 **:red[Tip]**- Check out more song options in the sidebar")

        elif selected_album is not None and len(albums) > 0:
            albums_list = albums['albums']['items']
            album_id = None
            album_uri = None
            if len(albums_list) > 0:
                for album in albums_list:
                    str_temp = album['name'] + "- By -" + album['artists'][0]['name']
                    if selected_album == str_temp:
                        album_id = album['id']
                        album_uri = album['uri']
                        album_name = album['name']
            if album_id is not None and album_uri is not None:
                st.write(f"Collecting all the tracks for the album: :orange[{album_name}]")
                album_tracks = sp.album_tracks(album_id)
                df_album_tracks = pd.DataFrame(album_tracks['items'])
                # st.dataframe(df_album_tracks)
                df_tracks_min = df_album_tracks.loc[:,['id', 'name', 'duration_ms', 'explicit', 'preview_url']]
                # st.dataframe(df_tracks_min)
                with st.container(border = True):
                    col01, col02, col03, col04 = st.columns((4, 2, 2, 4))
                    col01.write("**:red[Track name]**")
                    col02.write("**:red[Duration]**")
                    for index, idx in enumerate(df_tracks_min.index, start=1):
                        col1, col2, col3, col4 = st.columns((4, 2, 2, 4))
                        col11, col12 = st.columns((5, 5))
                        col1.write(f"{index}. **{df_tracks_min['name'][idx]}**")
                        def ms_to_min(input):
                           return round((input/60000),2)
                        min = ms_to_min(df_tracks_min['duration_ms'][idx])
                        col2.write(f"**{min} min**")
                        col3.write(f":red[Explicit?] {df_tracks_min['explicit'][idx]}")
                        col4.write(f":red[Spotify ID] {df_tracks_min['id'][idx]}")
                        if df_tracks_min['preview_url'][idx] is not None:
                            col11.audio(df_tracks_min['preview_url'][idx], format = 'audio/mp3')
                            col12.write(f"[preview link]({df_tracks_min['preview_url'][idx]})")
                            st.title("#")


        if selected_artist is not None and len(artist) > 0:
            artists_list = artists['artists']['items']
            artist_id = None
            artist_uri = None
            if len(artists_list) > 0:
                for artist in artists_list:
                    if selected_artist == artist['name']:
                        artist_id = artist['id']
                        artist_uri = artist['uri']

            if artist_id is not None:
                artist_choice = ['Albums 💿', 'Top Songs 🎵']
                selected_artist_choice = st.sidebar.selectbox('More from artist', artist_choice)

            if selected_artist_choice is not None:
                if selected_artist_choice == 'Albums 💿':
                    artist_uri = 'spotify:artist:' + artist_id
                    album_result = sp.artist_albums(artist_uri, album_type = 'album')
                    all_albums = album_result['items']
                    container = st.container(border = True)
                    with container:
                        col01, col02, col03 = st.columns((8, 7, 5))
                        col02.subheader('Album Information')
                        st.divider()
                        col1, col2, col3 = st.columns((6, 4, 2))
                        col1.write(':red[Album Names]')
                        col2.write(':red[Release Dates]')
                        col3.write(':red[Total tracks in album]')
                        # for album in all_albums:
                        for index, album in enumerate(all_albums, start=1):
                            with st.container():
                                col1, col2, col3 = st.columns((6, 4, 2))
                                col1.write(f"{index}. {album['name']}")
                                col2.write(album['release_date'])
                                col3.write(album['total_tracks'])
                elif selected_artist_choice == 'Top Songs 🎵':
                    col_df, col_plot = st.columns((5,5))
                    col_rec_list, col_graph = st.columns((5, 8))
                    st.write(f"Gathering top songs from artist: :orange[{selected_artist}]")
                    artist_uri = 'spotify:artist:' + artist_id
                    top_songs_result = sp.artist_top_tracks(artist_uri)
                    container = st.container(border = True)
                    with container:
                        col01, col02, col03 = st.columns((6, 7, 5))
                        col02.subheader(f"{selected_artist}'s Top Songs 🤩")
                        st.divider()
                        col01, col02, col03, col04 = st.columns((4, 2, 2, 4))
                        col04.write("*Song previews will populate if available*")
                        col01.write("#")
                        col01.write("**:red[Track Name]**")

                        for index, track in enumerate(top_songs_result['tracks'], start=1):
                            
                            col1, col2, col3, col4 = st.columns((5, 5, 5, 5))
                            col11, col12 = st.columns((5, 5))
                            col21, col22 = st.columns((5, 5))
                            col31, col32 = st.columns((5, 5))

                            col1.write(f"**{index}. {track['name']}**")

                            def feature_requested():
                                track_features = sp.audio_features(track['id'])
                                df2 = pd.DataFrame(track_features, index = [0])
                                df2_features = df2.loc[: ,['acousticness', 'danceability', 'energy',    'instrumentalness', 'liveness', 'speechiness', 'valence']]
                                with col_df:
                                    st.write(":red[Features]")
                                    st.dataframe(df2_features)
                                with col_plot:
                                    st.write(":red[Polarplot]")
                                    polarplot.feature_plot(df2_features)

                            feature_button_state = col2.button('Track Audio Features', key=track['id'],   on_click=feature_requested)

                    
                            with col3:
                                def similar_songs_requested():
                                    token = songrecommendations.get_token(SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET)
                                    similar_songs_json = songrecommendations.get_track_recommendations(track['id'], token)
                                    recommendation_list = similar_songs_json['tracks']
                                    recommendation_list_df = pd.DataFrame(recommendation_list)
                                    recommendation_df = recommendation_list_df[['name', 'explicit', 'duration_ms',  'popularity']]
                                    with col_rec_list:
                                        st.write("**:red[Similar Songs List]**")
                                        st.dataframe(recommendation_df, hide_index = True)
                                    with col_graph:
                                        st.write("**:red[Similar Songs Based on Explicitness, Duration, Popularity]**")
                                        st.write("Red circle means explicit. Size of circle based on popularity.")
                                        songrecommendations.song_recommendation_vis(recommendation_df)

                            similar_songs_state = col3.button('Similar Songs', key=track['name'],     on_click=similar_songs_requested)

                            col4.write(f":green[Spotify ID] {track['id']}")

                            if track['preview_url'] is not None:
                                col11.audio(track['preview_url'], format = 'audio/mp3')
                                col12.write(f"[preview link]({track['preview_url']})")
                            
            tip_container = st.container(border = True)
            tip_container.write("💡 **:red[Tip]**- Check out more from artist in the sidebar")

    st.sidebar.divider()
    st.sidebar.subheader(":violet[Audio] Definitions")
    definition_selected = st.sidebar.selectbox("Select a feature to define", definition_choices)

    with st.sidebar.expander("See definition"):
        if definition_selected is not None:
            st.write(definition(definition_selected))











if page == "The Song Popularity Dataset 📑":
    col01,col02,col03 = st.columns((1.5,5,1))
    col02.title(":rainbow[The Song Popularity Dataset] 🔢")
    st.divider()
    st.sidebar.divider()
    st.sidebar.subheader(":violet[Audio] Definitions")
    definition_selected = st.sidebar.selectbox("Select a feature to define", definition_choices)

    with st.sidebar.expander("See definition"):
        if definition_selected is not None:
            st.write(definition(definition_selected))

    col1, col2 = st.columns([3, 1])
    with col1:
        container = st.container(border= True)
        container.subheader(":blue[About the Data] 📝")
        container.write("The dataset encompasses numerous rows, each representing individual songs, with accompanying columns presenting detailed information for each song. This information includes the song's name, duration, popularity, and various musical attributes such as danceability, energy, valence, and others. *The original dataset was sourced from Kaggle, accessible [here](https://www.kaggle.com/datasets/yasserh/song-popularity-dataset/data?select=song_data.csv).* For convenience, the dataset can be obtained by selecting the `Download CSV` button!")
        # Download
        @st.cache_data
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
        csv = convert_df(df)
        container.download_button(
            label="Download CSV",
            data=csv,
            file_name='cleaned_song_data.csv',
            mime='text/csv')
        st.image("https://storage.googleapis.com/research-production/1/2021/10/Multi-Task-Graph-Learning-2.png", width = 700)
        
    with col2:
        def load_lottieurl(url: str):
            r = requests.get(url)
            if r.status_code != 200:
                return None
            return r.json()
        lottie_music_notes = load_lottieurl("https://lottie.host/101b7889-c559-400b-a6d3-41d3e02bf4ae/3mFOeOrgHn.json")
        st_lottie(lottie_music_notes, speed=1, height=200, width = 250, key='inital')

        lottie_data_overview = load_lottieurl("https://lottie.host/dc30209c-0aeb-47ef-8020-6ea11f527ac5/HDiUXa5jMP.json")        
        st_lottie(lottie_data_overview, speed=1, height=200)

        lottie_data_overview = load_lottieurl("https://lottie.host/b825f3b4-a7de-424b-96c4-8e7c7df111c3/HYjVWWaNCH.json")        
        st_lottie(lottie_data_overview, speed=0.75, height=200)

    container = st.container(border=True)
    container2 = st.container(border = True)

    with container:
        st.subheader(":red[Data Overview]")
        st.write("**Take a brief look at the Song Popularity Dataset provided below to get a glimpse of our current data!**")
        st.write("Please make your selection(s):")
        # Display dataset
        if container.checkbox("**:blue[Data Frame] :grey[(*the entire table*)]**"):
            container2.dataframe(df)
        # Column List
        if container.checkbox("**:green[Column List] :grey[(*names of the columns*)]**"):
            container2.code(f"Columns: {original_df.columns.tolist()}")
            if container.toggle('**:violet[Further breakdown of columns]**'):
                num_cols = df.select_dtypes(include = 'number').columns.tolist()
                obj_cols = df.select_dtypes(include = 'object').columns.tolist()
                container2.code(f"Numerical Columns: {num_cols} \nObject Columns: {obj_cols}")
        # Shape
        if container.checkbox("**:orange[Shape] :grey[(*aka # of rows and columns*)]**"):
            container2.write(f"**:grey[There are] :orange[{df.shape[0]}] rows :grey[and] :orange[{df.shape[1]}] columns.**")











# Build EDA page
if page == "Explore the Dataset 📊":

    st.sidebar.subheader(":violet[Audio] Definitions")
    definition_selected = st.sidebar.selectbox("Select a feature to define", definition_choices)

    with st.sidebar.expander("See definition"):
        if definition_selected is not None:
            st.write(definition(definition_selected))

    col1_spacer, col1, col2, col3_spacer = st.columns([0.1, 3, 2, 0.1])

    with col1:
        st.title("🎼 :blue[Exploratory Data Analysis] (EDA) 🔍")
        st.subheader("**Let your :rainbow[curiosity] take over!**")
        st.write("In this page, you can :blue[explore] the data by inputing :green[features] into the select boxes, which automatically generates different :orange[data visualizations] based on your selected choices!")
        container = st.container(border = True)
        with container:
            st.write("*The features are based on the features in The Song Popularity Dataset*")
            st.write("These features include:")
            col11, col12, col13 = st.columns((3, 3, 3,))
            col11.markdown('- Acousticness') 
            col11.markdown('- Danceability') 
            col11.markdown('- Energy') 
            col11.markdown('- Instrumentalness')
            col11.markdown('- Audio Mode')
            col12.markdown('- Key')
            col12.markdown('- Liveness') 
            col12.markdown('- Loudness')
            col12.markdown('- Speechiness') 
            col13.markdown('- Tempo') 
            col13.markdown('- Time Signature') 
            col13.markdown('- Valence')
            col13.markdown('- Song Duration')


    with col2:
        def load_lottieurl(url: str):
            r = requests.get(url)
            if r.status_code != 200:
                return None
            return r.json()  
        # lottie_eda = load_lottieurl("https://lottie.host/e60ee7ab-a894-45b8-b968-dcd08de24cd8/3VpqpZy1Hr.json") 
        lottie_eda = load_lottieurl("https://lottie.host/00e51377-d072-4c2f-b057-36f2f42477e1/MrZJPg6NCe.json")       
        st_lottie(lottie_eda, speed=1, height=260, key="initial")
        container2 = st.container(border = True)
        with container2:
            st.subheader(":blue[Define a feature] ♪")
            selected_feature = st.selectbox('Pick a feature you would like to define', ('Acousticness', 'Danceability', 'Energy', 'Instrumentalness', 'Key', 'Liveness', 'Loudness', 'Speechiness', 'Tempo', 'Time Signature', 'Valence', 'Song Duration'), index = 0, placeholder = "Choose an option")
            def define_feature(selected_feature):
                if selected_feature == 'Acousticness':
                    return f'A confidence measure from 0.0 to 1.0 of whether the track is acoustic. Acoustic means the music is created without the use of any electronic amplification or effects. It\'s just the pure, raw sound of the instruments. 1.0 represents high confidence the track is acoustic.'
                if selected_feature == 'Danceability':
                    return f'Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.'
                if selected_feature == 'Energy':
                    return f'Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.'
                if selected_feature == 'Instrumentalness':
                    return f'Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.'
                if selected_feature == 'Key':
                    return f'The key the track is in. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1.'
                if selected_feature == 'Liveness':
                    return f'Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.'
                if selected_feature == 'Loudness':
                    return f'The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typically range between -60 and 0 db.'
                if selected_feature == 'Speechiness':
                    return f'Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.'
                if selected_feature == 'Tempo':
                    return f'The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.'
                if selected_feature == 'Time Signature':
                    return f'A sign used to indicate musical meter and usually written with one number above another with the bottom number indicating the kind of note used as a unit of time and the top number indicating the number of these units in each measure.'
                if selected_feature == 'Valence':
                    return f'A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).'
                if selected_feature == 'Song Duration':
                    return f'THe amount of time an entire piece of work lasts'
                if selected_feature == 'Audio Mode':
                    return f'Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.'
            if selected_feature is not None:
                st.write(define_feature(selected_feature))


    

    st.divider()

    df2 = df.drop(columns = 'song_name')
    num_cols = df.select_dtypes(include = 'number').columns.tolist()
    obj_cols = df2.select_dtypes(include = 'object').columns.tolist()
    all_cols = df.select_dtypes(include = ['number','object']).columns.tolist()

    container = st.container(border=True)

    eda_type = container.multiselect("What types of graphs would you like to see? 📊", 
                              ['Histograms', 'Box Plots', 'Scatterplots', 'Count Plots'])





    # COUNTPLOTS
    if "Count Plots" in eda_type:
        st.subheader(":blue[Count Plots] :grey[-] Visualizing Relationships")
        
        col1, col2 = st.columns(2)
        col1, col2= st.columns([1, 3])
        col1.subheader("Selections ⌵")
        col2.subheader("Graphs 📊")

        with col1:
            container = st.container(border=True)
            c_selected_col = container.selectbox("Select a categorical column for your countplot:", obj_cols, index = None)
            if c_selected_col:
                c_selected_col2 = container.selectbox("Select another categorical column:", obj_cols, index = None)
                with col2:
                    container2 = st.container(border=True)
                    if c_selected_col2:
                        with col2:
                            chart_title = f"Count of {' '.join(c_selected_col.split('_')).title()} & {' '.join(c_selected_col2.split('_')).title()}"
                            # x_axis = f"{(c_selected_col.split('_')).title()}"
                            fig = container2.plotly_chart(px.bar(df, x = c_selected_col, title = chart_title, color = c_selected_col2, barmode='group'), use_container_width = True)
                    else:
                        with col2:
                            chart_title = f"Count of {' '.join(c_selected_col.split('_')).title()}"
                            fig = container2.plotly_chart(px.bar(df, x = c_selected_col, title = chart_title), use_container_width = True)
        st.write("---")




    # HISTOGRAMS
    if "Histograms" in eda_type:
        st.subheader(":violet[Histograms] :grey[-] Visualizing Numerical Distributions")
        
        col1, col2 = st.columns(2)
        col1, col2= st.columns([1, 3])
        col1.subheader("Selections ⌵")
        col2.subheader("Graphs 📊")

        with col1:
            container = st.container(border=True)
            h_selected_col = container.selectbox("Select a numerical column for your histogram:", num_cols, index = None)
            if h_selected_col:
                h_selected_col2 = container.selectbox("**Select a hue:**", obj_cols, index = None)
                with col2:
                    container2 = st.container(border=True)
                    if h_selected_col2:
                        with col2:
                            chart_title = f"Distribution of {' '.join(h_selected_col.split('_')).title()} Based On {' '.join(h_selected_col2.split('_')).title()}"
                            fig = container2.plotly_chart(px.histogram(df, x = h_selected_col, title = chart_title, barmode = 'overlay', color = h_selected_col2), use_container_width = True)
                    else:
                        with col2:
                            chart_title = f"**Distribution of {' '.join(h_selected_col.split('_')).title()}**"
                            fig = container2.plotly_chart(px.histogram(df, x = h_selected_col, title = chart_title), use_container_width = True)
        st.write("---")





    # BOXPLOTS
    if "Box Plots" in eda_type:
        st.subheader("**:green[Box Plots] :grey[-] Visualizing Numerical Distributions**")
        
        col1, col2 = st.columns(2)
        col1, col2= st.columns([1, 3])
        col1.subheader("Selections ⌵")
        col2.subheader("Graphs 📊")

        with col1:
            container = st.container(border=True)
            b_selected_col = container.selectbox("Select a numerical column for your box plot:", num_cols, index = None)
            if b_selected_col:
                b_selected_col2 = container.selectbox("**Select a hue:**", obj_cols, index = None)
                with col2:
                    container2 = st.container(border=True)
                    if b_selected_col2:
                        with col2:
                            chart_title = f"Distribution of {' '.join(b_selected_col.split('_')).title()} Based On {' '.join(b_selected_col2.split('_')).title()}"
                            fig = container2.plotly_chart(px.box(df, x = b_selected_col, y = b_selected_col2, title = chart_title, color = b_selected_col2), use_container_width = True)
                    else:
                        with col2:
                            chart_title = f"Distribution of {' '.join(b_selected_col.split('_')).title()}"
                            fig = container2.plotly_chart(px.box(df, x = b_selected_col, title = chart_title), use_container_width = True)
        st.write("---")





    # SCATTERPLOTS
    if "Scatterplots" in eda_type:
        st.subheader("**:orange[Scatterplots] :grey[-] Visualizing Relationships**")

        col1, col2 = st.columns(2)
        col1, col2= st.columns([1, 3])
        col1.subheader("Selections ⌵")
        col2.subheader("Graphs 📊")

        with col1:
            container = st.container(border=True)
            selected_col_x = container.selectbox("Select x-axis variable:", num_cols, index = None)
            selected_col_y = container.selectbox("Select y-axis variable:", num_cols, index = None)
            if selected_col_x and selected_col_y:
                selected_col_hue = container.selectbox("Select a hue:", obj_cols, index = None)
                with col2:
                    container2 = st.container(border=True)
                    chart_title = f"Relationship of {selected_col_x} vs. {selected_col_y}"
                    if selected_col_hue:
                        with col2:
                            container2.plotly_chart(px.scatter(df, x = selected_col_x, y = selected_col_y, title = chart_title, color = selected_col_hue, opacity = 0.5), use_container_width = True)
                    else:
                        with col2:
                            container2.plotly_chart(px.scatter(df, x = selected_col_x, y = selected_col_y, title = chart_title, opacity = 0.5), use_container_width = True)
        st.write("---")










# Build Modeling Page
if page == "Machine Learning Modeling 🤖":

    st.title("⚙️ How :orange[Machine Learning Modelling] Works 🤖")
    st.markdown("**On this page, you can see how well different :orange[machine learning models] make :violet[predictions] on 🎤 :red[song popularity]!**")
    st.write("There are two main types of Machine Learning Models: Machine Learning Classification (where the response belongs to a set of classes) and Machine Learning Regression(where the response is continuous). In this page, we can choose between three Machine Learning _Classification_ Models: :orange[_k_-Nearest Neighbor (KNN)], :blue[Logistic Regression], and :green[Random Forest]. **:grey[To learn about the different types of machine learning models, click the link found]** [here](https://www.mathworks.com/discovery/machine-learning-models.html#:~:text=There%20are%20two%20main%20types,where%20the%20response%20is%20continuous).")

    def load_lottieurl(url: str):
            r = requests.get(url)
            if r.status_code != 200:
                return None
            return r.json()  
    lottie_m = load_lottieurl("https://lottie.host/5685aad0-58cd-4833-acde-7f2967214db7/bbFj1U1YO9.json")        
    st_lottie(lottie_m, speed=1, height=200, key="initial")

    container = st.container(border = True)
    with container:
        col01, col02, col03 = st.columns((2, 5, 1))
        col02.subheader("Classification Machine Learning Models")
        left_spacer, col1, col1_spacer, col2, col2_spacer, col3, right_spacer = st.columns((0.5, 5, 0.5, 5, 0.5, 5, 0.5))
        col1.subheader(":orange[_k_-Nearest Neighbor]")
        col1.image("https://miro.medium.com/v2/resize:fit:1358/0*jqxx3-dJqFjXD6FA")
        with col1.expander("See description"):
            st.write("KNN is a type of machine learning model that categorizes objects based on the classes of their nearest neighbors in the data set. KNN predictions assume that objects near each other are similar. Distance metrics, such as Euclidean, city block, cosine, and Chebyshev, are used to find the nearest neighbor. Learn more [here](https://towardsdatascience.com/a-simple-introduction-to-k-nearest-neighbors-algorithm-b3519ed98e#:~:text=What%20is%20KNN%3F,how%20its%20neighbours%20are%20classified).")
            st.image("https://qph.fs.quoracdn.net/main-qimg-c6a1b46c814fd6347dc54750b57fefa8")
        col2.subheader(":blue[Logistic Regression]")
        col2.image("https://prwatech.in/blog/wp-content/uploads/2020/02/logi1.png")
        with col2.expander("See description"):
            st.write("Logistic regression is a special case of regression analysis and is used when the dependent variable is nominally scaled. With logistic regression, it is now possible to explain the dependent variable or estimate the probability of occurrence of the categories of the variable. Learn more [here](https://datatab.net/tutorial/logistic-regression).")
            st.image("https://miro.medium.com/v2/resize:fit:958/1*_4lc56CLCUtzgBCPxELJEA.jpeg")
        col3.subheader(":green[Random Forest]")
        col3.image("https://media5.datahacker.rs/2022/08/26.jpg")
        with col3.expander("See description"):
            st.write("A Random Forest is like a group decision-making team in machine learning. It combines the opinions of many “trees” (individual models) to make better predictions, creating a more robust and accurate overall model. It combines the output of multiple decision trees to reach a single result. Its ease of use and flexibility have fueled its adoption, as it handles both classification and regression problems. Learn more [here](https://www.ibm.com/topics/random-forest#:~:text=Random%20forest%20is%20a%20commonly,both%20classification%20and%20regression%20problems.).")
            st.image("https://pbs.twimg.com/media/FQJEPb6aUAQkbKP.jpg")

    container2 = st.container(border=True)
    container2.subheader("**The Baseline Score**") 
    container2.write(">The baseline score is often used as a starting point for evaluating the effectiveness of models or algorithms. It provides a clear indication of the minimum level of performance that needs to be exceeded to warrant the implementation of more complex and resource-intesive algorithms. These scores are obtained by adding the value of each of the samples together, then divide by the total number of samples. In Layman's terms, extracting the mean or average. If the score is lower than the mean, then there is no point in using the model.")
    container2.write(":violet[Low Popularity]: 12.65%")
    container2.write(":blue[Low-Mid Popularity]: 27.56%")
    container2.write(":orange[Mid-High Popularity]: 46.11%")
    container2.write(":red[High Popularity]: 13.68%")

    # Set up X and y
    X = df.drop(columns = ['song_name', 'song_popularity', 'popularity_category'])
    y = df['popularity_category']

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

    # model = RandomForestClassifier()

    container = st.container(border=True)
    model_option = container.selectbox(" Select a Model", ["KNN", "Logistic Regression", "Random Forest"], index = None)

    # Instantiating & fitting selected model
    if model_option:
        press_button = container.button("Let's see the performance!")
        if model_option == "KNN":
            k_value = container.slider("Select the number of neighbors (k)", 1, 29, 5, 2)
            model = KNeighborsClassifier(n_neighbors= k_value)
        elif model_option == "Logistic Regression":
            model = LogisticRegression()
        elif model_option == "Random Forest":
            model = RandomForestClassifier()
        if press_button:
            st.set_option('deprecation.showPyplotGlobalUse', False)
            model.fit(X_train, y_train)

            # Display results
            def model_colors(model):
                if model_option == 'KNN':
                    return f':orange[{model}]'
                elif model_option == 'Logistic Regression':
                    return f':blue[{model}]'
                elif model_option == 'Random Forest':
                    return f':green[{model}]'
                
            def use_image(score):
                if score > 46.11:
                    return container3.image('https://preview.redd.it/izsegksg4dr21.jpg?auto=webp&s=1808fbbe9d7c5667a95d4376b4757bbd269fbeb4', width = 300)
                else:
                    return container3.image('https://media.makeameme.org/created/dont-be-sad-901acc7b3d.jpg', width = 300)
                
            def use_model(score):
                if score > 46.11:
                    return f"🎉 The {model_colors(model)} Model beat the Baseline Score! You can use this model to make predictions for this dataset."
                else:
                    return f"🌧️ The {model_colors(model)} Model did not beat the basline score 🙅🏻‍♀️... You can probably use another model."

            def celebrate(score):
                if score > 46.11:
                    st.balloons()
                else:
                    st.snow()

            use_model_answer = use_model(round(model.score(X_test, y_test)*100, 2))
            container3 = st.container(border=True)
            container3.subheader(":blue[Performance]")
            container3.write(use_model_answer)
            time_to_use_image= use_image(round(model.score(X_test, y_test)*100, 2))
            time_to_celebrate = celebrate(round(model.score(X_test, y_test)*100, 2))


                
            container = st.container(border=True)
            container.subheader(":blue[Evaluation]")
            container.text(f"Accuracy score on training dataset: {round(model.score(X_train, y_train)*100, 2)}%")
            container.write(f"**:orange[Accuracy score on testing dataset]: :red[{round(model.score(X_test, y_test)*100, 2)}%] ← ------ The testing score is compared to :orange[Mid-High Popularity] baseline score.**")
            container.image("https://pbs.twimg.com/media/EegSVtOXkAAcI_l.jpg", width = 300)

            # Confusion Matrix
            st.subheader(f"Confusion Matrix Using {model_colors(model)} Model")
            ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap = 'Blues', xticks_rotation=45)
            st.pyplot()












# Build Predictions Page
if page == "Predict Song Popularity 🔮":

    st.sidebar.divider()
    st.sidebar.subheader(":violet[Audio] Definitions")
    definition_selected = st.sidebar.selectbox("Select a feature to define", definition_choices)

    with st.sidebar.expander("See definition"):
        if definition_selected is not None:
            st.write(definition(definition_selected))

    description_choices = ['k-Nearest Neighbor', 'Logistic Regression', 'Random Forest']

    def description(description_choices):
        if description_choices == 'k-Nearest Neighbor':
            return f'KNN is a type of machine learning model that categorizes objects based on the classes of their nearest neighbors in the data set. KNN predictions assume that objects near each other are similar. Distance metrics, such as Euclidean, city block, cosine, and Chebyshev, are used to find the nearest neighbor. Learn more [here](https://towardsdatascience.com/a-simple-introduction-to-k-nearest-neighbors-algorithm-b3519ed98e#:~:text=What%20is%20KNN%3F,how%20its%20neighbours%20are%20classified).' 
        if description_choices == 'Logistic Regression':
            return f'Logistic regression is a special case of regression analysis and is used when the dependent variable is nominally scaled. With logistic regression, it is now possible to explain the dependent variable or estimate the probability of occurrence of the categories of the variable. Learn more [here](https://datatab.net/tutorial/logistic-regression).'
        if description_choices == 'Random Forest':
            return f'A Random Forest is like a group decision-making team in machine learning. It combines the opinions of many “trees” (individual models) to make better predictions, creating a more robust and accurate overall model. It combines the output of multiple decision trees to reach a single result. Its ease of use and flexibility have fueled its adoption, as it handles both classification and regression problems. Learn more [here](https://www.ibm.com/topics/random-forest#:~:text=Random%20forest%20is%20a%20commonly,both%20classification%20and%20regression%20problems.).'
    def images(description_choices):
        if description_choices == 'k-Nearest Neighbor':
            return f"https://qph.fs.quoracdn.net/main-qimg-c6a1b46c814fd6347dc54750b57fefa8"
        if description_choices == 'Logistic Regression':
            return f"https://miro.medium.com/v2/resize:fit:958/1*_4lc56CLCUtzgBCPxELJEA.jpeg"
        if description_choices == 'Random Forest':
            return f"https://pbs.twimg.com/media/FQJEPb6aUAQkbKP.jpg"
            
    st.sidebar.divider()
    st.sidebar.subheader(":blue[Machine Learning Models] Descriptions")
    description_selected = st.sidebar.selectbox("Select a model to describe", description_choices)

    with st.sidebar.expander("See description"):
        if description_selected is not None:
            st.write(description(description_selected))
            st.image(images(description_selected))


    st.title("♪ :violet[Predictions] 🔮")
    st.markdown("On this page, you can make :violet[predictions] as to which :red[popularity category] a song will fit in based on features contained in the dataset using the :orange[Machine Learning Classification Model] of your choice!")
    def load_lottieurl(url: str):
            r = requests.get(url)
            if r.status_code != 200:
                return None
            return r.json()    
       
    lottie_pred = load_lottieurl("https://lottie.host/f397d1bb-9122-45d7-9d35-85fa5e8034ed/ZFCCFCYBw8.json")        
    st_lottie(lottie_pred, speed=1, height=300, key="initial")

    def model_colors(model):
        if model_option == 'KNN':
            return f':orange[{model}]'
        elif model_option == 'Logistic Regression':
            return f':blue[{model}]'
        elif model_option == 'Random Forest':
            return f':green[{model}]'


    container = st.container(border=True)

    container2 = st.container(border = True)
    container2.subheader(":rainbow[Get Inspiration] or Compare Your Values to Other Songs")
    container2.dataframe(df, hide_index = True)

    with container:
        col01,col02,col03 = st.columns((5,5,4))
        col02.subheader("Input your values 📝")
        st.write("*Default values from :orange['Mr.Brightside'] By The Killers*")
        col1, col2 = st.columns((5,5))

        # age_num = container.number_input("What\'s the age of the passenger? Pick an age from 1 to 100", min_value = 1, max_value = 100, step = 1, value=None, placeholder="Type a number...")
        # if age_num:
        #     container2.write(f'The age is {age_num}')
        # else:
        #     container2.write(":red[Please enter the age]")

        acousticness = col1.number_input("Acousticness 🎸 (0.0 to 100.0)", min_value = 0.0, max_value = 100.0, step = 1.0, 
        value=0.00108, placeholder="Type a number...")
            
        danceability = col2.number_input("Danceability 🕺🏼 (0.0 to 1.0)", min_value = 0.0, max_value = 1.0, step = .01, value=0.33, placeholder="Type a number...")

        energy = col2.number_input("Energy ⚡️ (Input a number between 0.0 to 1.0)", min_value = 0.0, max_value = 1.0, step = .01, value=0.936, placeholder="Type a number...")

        instrumentalness = col1.number_input("Instrumentalness 🎷(0.0 to 1.0)", min_value = 0.0, max_value = 1.0, step = .01, value=0.01, placeholder="Type a number...")

        key = col2.number_input("Key ♪ (0 to 11)", min_value = 0, max_value = 11, step = 1, value=1, placeholder="Type a number...")

        liveness = col1.number_input("Liveness 👩🏻‍🎤(0.0 to 1.0)", min_value = 0.0, max_value = 1.0, step = .01, value=0.0926, placeholder="Type a number...")    

        loudness = col2.number_input("Loudness 🔊 (-40.0 to 2.0)", min_value = -60.0, max_value = 0.0, step = 1.0, value=-3.66, placeholder="Type a number...")     

        audio_mode = col1.radio("What's the audio mode? 🎙️",
        [0, 1], index=1,)

        speechiness = col2.number_input("Speechiness 🗣️ (0.0 to 1.0)", min_value = 0.0, max_value = 1.0, step = .01, value=0.0917, placeholder="Type a number...")    

        tempo = col1.number_input("Tempo 🥁 (1 to 250)", min_value = 1, max_value = 250, step = 1, value=148, placeholder="Type a number...")   

        time_signature = col2.number_input("Time Signature 🎼 (0 to 5)", min_value = 0, max_value = 5, step = 1, value=4, placeholder="Type a number...")  

        audio_valence  = col1.number_input("Valence 🪩 (0.0 to 1.0)", min_value = 0.0, max_value = 1.0, step = .01, value=0.234, placeholder="Type a number...")

        song_duration_min =col2.number_input("What is the duration of the song in minutes? ⏳ (Max 30 min)", min_value = 0.1, max_value = 30.00, step = 0.1, value=3.71, placeholder="Type a number...") 

        # Your features must be in order that the model was trained on
        user_input = pd.DataFrame({
                    'acousticness':[acousticness],
                    'danceability':[danceability],
                    'energy':[energy],
                    'instrumentalness':[instrumentalness],
                    'key':[key],
                    'liveness':[liveness],
                    'loudness':[loudness],
                    'audio_mode':[audio_mode],
                    'speechiness':[speechiness],
                    'tempo':[tempo],
                    'time_signature':[time_signature],
                    'audio_valence':[audio_valence],
                    'song_duration_min':[song_duration_min]})


    # Fitting a model
    X = df.drop(columns = ['song_name', 'song_popularity', 'popularity_category'])
    y = df['popularity_category']


    # if user_input is not None:
    # Model Selection
    container2= st.container(border=True)
    container2.subheader(":orange[Choose a Machine Learning Classification Model] to use to make your :violet[prediction]")
    container2.write("> To get a refresher on the different Machine Learning models, check the `sidebar` for their descriptions.")
    model_option = container2.selectbox("Please choose a model to use", ["Random Forest", "Logistic Regression", "KNN"])
    
    if model_option:
        container2.write(f'You selected: {model_colors(model_option)}')
        # Instantiating & fitting selected model
        if model_option == "KNN":
            k_value = container2.slider("Select the number of neighbors (k)", 1, 21, 5, 2)
            model = KNeighborsClassifier(n_neighbors=k_value)
        elif model_option == "Logistic Regression":
            model = LogisticRegression()
        if model_option == "Random Forest":
            model = RandomForestClassifier()
        
        # def make_prediction():
            # msg = st.toast('Gathering the data...')
            # time.sleep(1)
            # msg.toast('Analyzing...')
            # time.sleep(1)
            # msg.toast('Ready!', icon = "🔮")
        
        

        if container2.button("Make Your Prediction"):
            with st.spinner('Wait for it...'):
                model.fit(X, y)
                prediction = model.predict(user_input)
                st.success('The results are in!')
        # Popularity category colors
            def category_colors(prediction):
                if prediction == 'Low Popularity':
                    return f':violet{prediction}'
                elif prediction == 'Low-Mid Popularity':
                    return f':blue{prediction}'
                elif prediction == 'Mid-High Popularity':
                    return f':orange{prediction}'
                elif prediction == 'High Popularity':
                    return f':red{prediction}'
        # make_prediction()
            if prediction == "Low Popularity" or prediction == "Low-Mid Popularity":
                container2.subheader(f"{model_colors(model_option)} predicts that this song will have {category_colors(prediction)}. Try inputting different values in certain features to see if the song will become more popular. Or try using a different Machine Learning Model.")
                container2.image('https://i.imgflip.com/kfmzg.jpg', width = 500)
                st.snow()
            elif prediction == "Mid-High Popularity" or prediction == "High Popularity":
                container2.subheader(f"{model_colors(model_option)} predicts that this song will have ✨{category_colors(prediction)} ✨! :rainbow[Congratulations]!")
                container2.image('https://i.redd.it/rc6bq0lmxka51.jpg', width = 500)
                st.balloons()
            else:
                container2.header(":red[Missing inputs]. Please recheck your inputs.")

st.sidebar.divider()
st.sidebar.write("Created wtih ❤️ by [Ann](https://www.linkedin.com/in/ann-daniel-celestino-459333184)")
st.sidebar.write("#")
st.sidebar.write("#")
st.sidebar.write("#")
with st.sidebar:
    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    lottie_data_overview = load_lottieurl("https://lottie.host/7c362cab-b723-4f58-9885-609e3b02a039/dN1W72TpU9.json")        
    st_lottie(lottie_data_overview, speed=.70, height=200)

st.divider()
st.write("🔧 Last Updated: January 16, 2024")