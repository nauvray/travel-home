import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import requests



st.sidebar.markdown("Predict your **holidays** in **France**")


######################
# Titre de l'application
######################
st.sidebar.markdown("## TRAVEL-HOME")

img_test_path1 = 'app/images/Wagon_love.png'
img_pres = Image.open('app/images/TravelHomeGoogle.png')
st.image(img_pres, caption='', use_column_width=True)
