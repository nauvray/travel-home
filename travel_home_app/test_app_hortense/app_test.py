import streamlit as st
import requests
from PIL import Image
import pandas as pd
from streamlit_folium import st_folium
from folium import folium
from io import BytesIO
from utils import get_map, random_images, launch_plexel
from st_clickable_images import clickable_images
import subprocess

# LOGO AND BACKGROUND
#st.set_page_config(page_icon="logo.png", page_title="Travel Home", layout="wide")
st.set_page_config(page_title="Travel Home", layout="wide")

def add_bg_from_local():
    st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(to right, #C7F1AF, #F9EA8F);
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local()

# TITLE AND SUBTITLE

# image = Image.open('logo2.png')
# st.image(image, width=120)

title = '<p style="font-family:Cooper Black; color:Black; font-size: 50px">Travel Home</p>'
st.markdown(title, unsafe_allow_html=True)

sub_title = '<p style="font-family:Helvetica, color:Black; font-size: 30px">Find your dream travel destination at a train distance</p>'
st.markdown(sub_title, unsafe_allow_html=True)

st.write(" ")
st.write(" ")
st.write(" ")

# DATA INPUT FOR PREDICT OF API
columns = st.columns([7,3])
selected = columns[0].text_input("Enter your dream destination (or just a keyword 😉)", "")

get_prediction = False
if selected: # input = bar de recherche
    list_link=launch_plexel(selected)
    st.markdown("Now, click on your favorite image 👇")
    clicked = clickable_images(
    list_link,
    titles=[f"Image #{str(i)}" for i in range(6)],
    div_style={"display": "flex", "flex-wrap": "wrap", "justify-content": "center", "background-color" : "transparent", "width": "100%"},
    img_style={"margin": "1.5%", "height": "200px"},
    )
    if clicked >-1:
        get_prediction = True

image_uploaded = columns[1].file_uploader('or upload a picture', type=['jpg', 'jpeg', 'png']) # input = image upload

image = ''
if get_prediction==True:
    if image_uploaded is None :
        response = requests.get(list_link[clicked])
        st.write(list_link[clicked])
        image = Image.open(BytesIO(response.content))
        image.save('image.jpg')
        image = 'image.jpg'
    else :
        image = image_uploaded

# API
if get_prediction == True:
    url = 'https://travel-home-mzfiw6j4fa-ew.a.run.app/predict'
    #params = {'image': f'../../travel_home_app/test_app_hortense/{image}'}
    #st.write(params)
    request = requests.get(f'{url}?image=../../travel_home_app/test_app_hortense/{image}')
    #request = requests.get(url, params=params)
    st.write(request.url)
    data = request.json()
    df = pd.DataFrame(data, dtype='object')


# output api test en attendant l'api
data_test = {'cellid': [5169846499198107648, 5171065032959590400, 5220109917347643392],
        'probability': [0.92, 0.74, 0.34]}
df_test = pd.DataFrame(data_test, dtype='object')


if get_prediction == True:
    col1, col2, col3 = st.columns([5,3,1])
    with col1:
        st_folium(get_map(df), width=700, height=500)
    with col2:
        st.write("")
    with col3:
        st.write("")
    # DETAILS OF DESTINATION
    france_zoom=5
    with st.expander("Click here for the details of the destinations we found for you"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Destination 1", f"{df['new_proba'][0]}% of similarities")
            map_1 = folium.Map(location=(df['lat'][0], df['lon'][0]),
            zoom_start=france_zoom+2,zoom_control=False)
            show_map_1 = st_folium(map_1,width=400,height=150)
            # random_images(int(df['cellid'][0]))
            # columns = st.columns(2)
            # columns[0].image('proposal_0.jpg')
            # columns[1].image('proposal_1.jpg')
        with col2:
            st.metric("Destination 2", f"{df['new_proba'][1]}% of similarities")
            map_1 = folium.Map(location=(df['lat'][1], df['lon'][1]),
            zoom_start=france_zoom+2,zoom_control=False)
            show_map_1 = st_folium(map_1,width=400,height=150)
            # random_images(int(df['cellid'][1]))
            # columns = st.columns(2)
            # columns[0].image('proposal_0.jpg')
            # columns[1].image('proposal_1.jpg')
        with col3:
            st.metric("Destination 3", f"{df['new_proba'][2]}% of similarities")
            map_1 = folium.Map(location=(df['lat'][2], df['lon'][2]),
            zoom_start=france_zoom+2,zoom_control=False)
            show_map_1 = st_folium(map_1,width=400,height=150)
            # random_images(int(df['cellid'][2]))
            # columns = st.columns(2)
            # columns[0].image('proposal_0.jpg')
            # columns[1].image('proposal_1.jpg')
