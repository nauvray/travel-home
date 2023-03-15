import streamlit as st
import requests
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from travel_home.mapping.api import launch_plexel, clickable_images
from streamlit_folium import st_folium
from folium import folium, CircleMarker
from folium import Circle
import s2cell
from geopy.geocoders import Nominatim
from io import BytesIO

# logo, nom de la page et background
st.set_page_config(page_icon="logo.png", page_title="Travel Home", layout="wide")
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

# titre, sous-titre
image = Image.open('logo2.png')
st.image(image, width=120)

title = '<p style="font-family:Cooper Black; color:Black; font-size: 50px">Travel Home</p>'
st.markdown(title, unsafe_allow_html=True)

sub_title = '<p style="color:#FFFFF; font-size: 25px">Find your dream travel destination at a train distance 🚃</p>'
st.markdown(sub_title, unsafe_allow_html=True)

st.write(" ")
st.write(" ")
st.write(" ")

# bar de recherche et/ou upload de l'image
columns = st.columns([7,3])
selected = columns[0].text_input("Enter your dream destination (or just a keyword 😉)", "")

get_prediction = False

if selected:
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

image_uploaded = columns[1].file_uploader('or upload a picture', type=['jpg', 'jpeg', 'png'])

if image_uploaded is None and len(selected)>0:
    response = requests.get(list_link[clicked])
    image = Image.open(BytesIO(response.content))
    image.save('image.jpg')
    image = 'image.jpg'
else :
    image = image_uploaded

# avoir la map
def get_map(df_test):
    # create new dataframe with center and % of weight
    df_test[['lat','lon']] = df_test.apply(lambda x: s2cell.cell_id_to_lat_lon(x.cellid), axis=1, result_type='expand')
    # create a new column = weight in %
    df_test['new_weight'] = df_test['weight'].apply(lambda x: round(x*100))

    threshold = 10
    df_select = df_test[df_test.new_weight > threshold]
    geolocator = Nominatim(user_agent="my-app")
    location = geolocator.geocode("France")
    map_fr = folium.Map(location=[location.latitude, location.longitude], zoom_start=6)

    for lat, lon, weight in zip(df_select['lat'], df_select['lon'], df_select['new_weight']):
        CircleMarker(location=[lat, lon],
                    radius=weight/2,
                    color='blue',
                    fill=True,
                    fill_color='blue').add_to(map_fr)

    return map_fr

###api
# url = 'https://taxifare.lewagon.ai/predict' #api
# response = requests.get(url, params={image=image}).json()
# dico_pred = predict(image)
# df = pd.Dataframe(dicto_pred).from_dict


# output api test en attendant l'api
data = {'cellid': [5180546946359623680, 5189765251846897664, 5180949848651726848],
        'weight': [0.92, 0.74, 0.34]}
df_test = pd.DataFrame(data, dtype='object')

#st_folium(get_map(df_test), width=700, height=500)

if get_prediction == True:
    col1, col2, col3 = st.columns([5,3,1])
    with col1:
        st_folium(get_map(df_test), width=700, height=500)
    with col2:
        st.write("")
    with col3:
        st.write("")

    #details on destinations
    france_zoom=5
    with st.expander("Click here for the details of the destinations we found"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Destination 1", f"{df_test['new_weight'][0]}% of similarities")
            map_1 = folium.Map(location=(df_test['lat'][0], df_test['lon'][0]),
            zoom_start=france_zoom+2,zoom_control=False)
            show_map_1 = st_folium(map_1,width=400,height=150)
        with col2:
            st.metric("Destination 2", f"{df_test['new_weight'][1]}% of similarities")
            map_1 = folium.Map(location=(df_test['lat'][1], df_test['lon'][1]),
            zoom_start=france_zoom+2,zoom_control=False)
            show_map_1 = st_folium(map_1,width=400,height=150)
        with col3:
            st.metric("Destination 3", f"{df_test['new_weight'][2]}% of similarities")
            map_1 = folium.Map(location=(df_test['lat'][2], df_test['lon'][2]),
            zoom_start=france_zoom+2,zoom_control=False)
            show_map_1 = st_folium(map_1,width=400,height=150)









# def add_header_from_local():
#     st.markdown(
#     f"""
#     <style>
#     .stheader {{
#         background-color:#FF5757);
#     }}
#     </style>
#     """,
#     unsafe_allow_html=True
#     )
# add_header_from_local()

# html=("""<div class="header">
#   <h1>Header</h1>
#   <p>My supercool header</p>
# </div>""")

# st.components.v1.html(html, width=None, height=None, scrolling=False)
