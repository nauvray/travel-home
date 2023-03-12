import streamlit as st
# Add to requirements
from streamlit_folium import st_folium
# Add to requirements
#Add folium to requirements
from folium import folium
from folium import plugins
from folium import Circle,Marker
from folium.features import DivIcon

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)

def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)

local_css("style.css")
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

icon("search")
selected = st.text_input("", "Search...")
button_clicked = st.button("OK")

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
    st.header("Offer 1",)
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
