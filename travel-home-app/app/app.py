import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import requests
from io import BytesIO

# commande terminal =  streamlit run app/app.py




######################
# Titre de l'application
######################
st.sidebar.markdown("## TRAVEL-HOME")

img_test_path1 = 'app/images/Wagon_love.png'
img_pres = Image.open('app/images/TravelHomeGoogle.png')
st.image(img_pres, caption='', use_column_width=True)


st.write('')
st.write('')
st.header(':blue[**_Dream travel destination at a train/car distance_**]')

st.write('')


uploaded_file = st.file_uploader('**Please, choose one picture**', type=['jpg', 'jpeg', 'png'])


#########################
# insertion logo
#########################

logo = "app/images/LogoGoogleSust1.png"


# Créer un conteneur
with st.container():

# Ajouter le logo dans le conteneur
    st.image(logo, width=60)


if uploaded_file is not None:
    ### Display the image user uploaded

    image_uploaded = Image.open(uploaded_file)
    st.image(image_uploaded, caption='Your Dream Destination', use_column_width=True)

    st.write('')
    st.header(':green[**_Look at similar places in France_**]')
    st.write('')


#########################
# Prediction Points sur la carte de France
#########################

    # --> Insérer fonction bubble_plot de return.py

    img_test_path2 = 'app/images/france.png'
    img_test2 = Image.open(img_test_path2)


    st.image(img_test2, caption='', use_column_width=True)


#########################
# images de lieux similaires en France
#########################

    # --> Insérer fonction plot_4pics_around de return.py

# Créer quatre colonnes d'égale largeur
    col1, col2, col3, col4 = st.columns(4)

# Ajouter du contenu à la première colonne
    with col1:



        st.write('')

        img_test_path1 = 'app/images/bretagne.png'
        img_test1 = Image.open(img_test_path1)
        st.image(img_test1, caption='Your next french holidays', use_column_width=True)

# Ajouter du contenu à la deuxième colonne
    with col2:

        st.write('')

        img_test_path2 = 'app/images/bretagne.png'
        img_test2 = Image.open(img_test_path2)
        st.image(img_test2, caption='Your next french holidays', use_column_width=True)

# Ajouter du contenu à la 3ème colonne
    with col3:

        st.write('')

        img_test_path2 = 'app/images/bretagne.png'
        img_test2 = Image.open(img_test_path2)
        st.image(img_test2, caption='Your next french holidays', use_column_width=True)

# Ajouter du contenu à la 4ème colonne
    with col4:

        st.write('')

        img_test_path2 = 'app/images/bretagne.png'
        img_test2 = Image.open(img_test_path2)
        st.image(img_test2, caption='Your next french holidays', use_column_width=True)



    ### Get hexa from the upload file
    #image_hexa = Image.open(BytesIO(bytes.fromhex(uploaded_file)))


    # params = dict(image = image_hexa)
    #travel_home_api_url = 'https://XXXXXXXXXX/predict'

    ### Make request to  API
    #response = requests.get(travel_home_api_url, params=params)


    #if response.status_code == 200:
        #### Display taggs returned by the API
        # prediction = response.json()

        # pred = prediction['image_hexa']

        # st.header(f'')


    #else:
        # st.markdown("**Oops**, something went wrong 😓 Please try again.")
