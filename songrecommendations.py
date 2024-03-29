import requests
import base64

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
from PIL import Image

def get_token(clientId,clientSecret):
    url = "https://accounts.spotify.com/api/token"
    headers = {}
    data = {}
    message = f"{clientId}:{clientSecret}"
    messageBytes = message.encode('ascii')
    base64Bytes = base64.b64encode(messageBytes)
    base64Message = base64Bytes.decode('ascii')
    headers['Authorization'] = "Basic " + base64Message
    data['grant_type'] = "client_credentials"
    r = requests.post(url, headers=headers, data=data)
    token = r.json()['access_token']
    return token

def get_track_recommendations(seed_tracks,token):
    limit = 10
    recUrl = f"https://api.spotify.com/v1/recommendations?limit={limit}&seed_tracks={seed_tracks}"

    headers = {
        "Authorization": "Bearer " + token
    }

    res = requests.get(url=recUrl, headers=headers)
    return res.json()

def song_recommendation_vis(reco_df):
    
    reco_df['duration_min'] = round(reco_df['duration_ms'] / 60000, 1)
    reco_df["popularity_range"] = reco_df["popularity"] - (reco_df['popularity'].min() - 1)

    plt.figure(figsize=(12, 7), facecolor= "mistyrose")    

    y = reco_df['name']
    x = reco_df['duration_min']
    s = reco_df['popularity_range']*20

    color_labels = reco_df['explicit'].unique()
    rgb_values = sns.color_palette("Set2", 8)
    color_map = dict(zip(color_labels, rgb_values))

    plt.scatter(x, y, s, alpha=0.7, c=reco_df['explicit'].map(color_map))
    plt.xticks(fontsize= 15)
    plt.yticks(fontsize = 15)
    plt.title('Similar Songs Based on Duration, Popularity, and Explicitness', fontsize = 24)
    plt.ylabel('Similar Songs', fontsize = 15)
    plt.xlabel('Song Duration in Minutes', fontsize = 15)

    # show the graph
    plt.show()

    st.pyplot(plt)


def save_album_image(img_url, track_id):
    r = requests.get(img_url)
    open('images/' + track_id + '.jpg', 'wb').write(r.content)
    

def get_album_image(track_id):
    return Image.open('images/' + track_id + '.jpg')