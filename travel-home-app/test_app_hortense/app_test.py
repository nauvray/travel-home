import streamlit as st
import requests
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path

# with open('style.css') as f:
#     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

#a remettre
st.set_page_config(page_icon="logo3.png", page_title="Travel Home", layout="wide")
# image = Image.open('automne.jpg')
# st.image(image, width=1200)

image = Image.open('logo4.png')
st.image(image, width=200)

original_title = '<p style="font-family:Cooper Black; color:#FF5757; font-size: 50px;">Travel Home</p>'
st.markdown(original_title, unsafe_allow_html=True)




new_title = '<p style="color:Black; font-size: 20px;">Dream travel destination at a train/car distance</p>'
st.markdown(new_title, unsafe_allow_html=True)

columns = st.columns(2)
text_1 = columns[0].text_input("Enter your dream destination...")
upload = columns[1].file_uploader('or upload a picture', type=['jpg', 'jpeg', 'png'])


col1, col2, col3 = st.columns(3)
col1.metric("Destination 1", "Bretagne")
col2.metric("Destination 2", "Côte d'Azur")
col3.metric("Destination 3", "Pyrénées")

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
