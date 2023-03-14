import streamlit as st
# Add to requirements
from streamlit_folium import st_folium
# Add to requirements
#Add folium to requirements
from folium import folium
from folium import plugins
from folium import Circle,Marker
from folium.features import DivIcon
import numpy as np
import requests
from st_clickable_images import clickable_images
from PIL import Image

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)

def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)

def launch_plexel(word:str):
    photo = 'https://api.unsplash.com/search/photos'
    my_params = {'query':word,'client_id':'7ANOoawIlsbj-XMwK6am_kjkYwN_w-TnsNfgz0aKHFU'}
    x = requests.get(photo,params=my_params)
    x.json()
    list_link=[]
    for i in range(len(x.json()['results'])):
        if i < 5 and x.json()['results'][i]['urls']['small'] not in list_link:
            list_link.append(x.json()['results'][i]['urls']['small'])

    # if len(x.json()['results'])==0:
    #     'Error'
    # elif len(x.json()['results'])<5:
    #     random_list = np.random.randint(0,len(x.json()['results']),len(x.json()['results']))
    #     for i in random_list:
    #         list_link.append(x.json()['results'][i]['urls']['small'])
    # elif len(x.json()['results'])>=5:
    #     random_list = np.random.randint(0,len(x.json()['results']),5)
    #     for i in random_list:
    #         list_link.append(x.json()['results'][i]['urls']['small'])
    return list_link


image = Image.open('travel_home_logo.png')

st.image(image, caption='your dream travel destination at a train/car distance')

local_css("style.css")
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

icon("search")
selected = st.text_input("")
# button_clicked = st.button("OK")

if selected:
    list_link=launch_plexel(selected)
    clicked = clickable_images(
    list_link,
    titles=[f"Image #{str(i)}" for i in range(5)],
    div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
    img_style={"margin": "5px", "height": "200px"},
    )
    if clicked>-1:
        st.markdown("Partez en Bretagne c'est le centre du monde")

    # st.markdown(f"Image #{clicked} clicked" if clicked > -1 else "No image clicked")
    # button_clicked = st.button("Va chercher")

    # if len(list_link)==1:
    #     st.image(list_link[0])
    #     st.checkbox(label='',key='chkbx_1',)
    # elif len(list_link)==2:
    #     col1,col2=st.columns(2)
    #     with col1:
    #         st.image(list_link[0])
    #         st.checkbox(label='',key='chkbx_1')
    #     with col2:
    #         st.image(list_link[1])
    #         st.checkbox(label='',key='chkbx_2')
    # elif len(list_link)>2:
    #     col1,col2,col3=st.columns(3)
    #     with col1:
    #         st.image(list_link[0])
    #         st.checkbox(label='',key='chkbx_1')
    #     with col2:
    #         st.image(list_link[1])
    #         st.checkbox(label='',key='chkbx_2')
    #     with col3:
    #         st.image(list_link[2])
    #         st.checkbox(label='',key='chkbx_3')

        nb_results=3

        france_location =(47,2)
        france_zoom=5
        big_map = folium.Map(location=france_location,zoom_start=france_zoom,min_lat=-5,min_lon=42,max_lat=10,max_lon=52)

        choice_1_lon = 43.6
        choice_1_lat= -1.5
        choice_2_lon = 47
        choice_2_lat= -2
        choice_3_lon = 43.4
        choice_3_lat= 6.8

        choice_1_proba=80
        choice_2_proba=55
        choice_3_proba=45

        Circle(location=[choice_1_lon, choice_1_lat], tooltip='Point 1',
                fill_color='#000', radius=choice_1_proba*1000,
                weight=2, color="#000").add_to(big_map)
        Circle(location=[choice_2_lon, choice_2_lat], tooltip='Point 2',
                fill_color='#000', radius=choice_2_proba*1000,
                weight=2, color="#000").add_to(big_map)
        Circle(location=[choice_3_lon, choice_3_lat], tooltip='Point 3',
                fill_color='#000', radius=choice_3_proba*1000,
                weight=2, color="#000").add_to(big_map)

        st_bigmap = st_folium(big_map,width=700,height=450)

        col1,col2,col3=st.columns(nb_results)
        col1,col2,col3=st.columns(nb_results)
        with col1:
            st.header("Offer 1")
            map_1 = folium.Map(location=(choice_1_lon,choice_1_lat),
            zoom_start=france_zoom+5,zoom_control=False)
            show_map_1 = st_folium(map_1,width=200,height=150)
        with col2:
            st.header("Offer 2")
            map_2 = folium.Map(location=(choice_2_lon,choice_2_lat),
            zoom_start=france_zoom+5,zoom_control=False)
            show_map_2 = st_folium(map_2,width=200,height=150)
        with col3:
            st.header("Offer 3")
            map_3 = folium.Map(location=(choice_3_lon,choice_3_lat),
            zoom_start=france_zoom+5,zoom_control=False)
            show_map_3 = st_folium(map_3,width=200,height=150)
