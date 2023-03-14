import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
from return3.return3 import bubble_plot
from streamlit_folium import st_folium
import os
# commande terminal =  streamlit run app/app.py


######################
# Insertion Bandeau du haut
######################

bandeau = Image.open('app/images/TravelHome3.png')
st.image(bandeau, caption='', use_column_width=True)

######################
# Titre de l'application
######################
st.sidebar.markdown("## TRAVEL-HOME")


#img_pres = Image.open('app/images/TravelHomeGoogle.png')
#st.image(img_pres, caption='', use_column_width=True)


title = "Dream travel destination at a train/car distance"
center_title = f"<center><h1>{title}</h1></center>"
st.markdown(center_title, unsafe_allow_html=True)

st.write('')

col1, col2 = st.columns(2)
with col1:




with col2:




#########################
# insertion logo
#########################

logo = "app/images/LogoGoogleSust1.png"


# Créer un conteneur
with st.container():
# Ajouter le logo dans le conteneur
    st.image(logo, width=60)



uploaded_file = st.file_uploader('**Please, upload your picture**', type=['jpg', 'jpeg', 'png'])

# Si une image a été téléchargée, afficher l'image en utilisant st.image
if uploaded_file is not None:

    image_uploaded = Image.open(uploaded_file)
    st.image(image_uploaded, caption='Your Dream Destination', use_column_width=True)


    texte = "Look at similar places in France"
    center_texte = f"<center><h2>{texte}</h2></center>"

    st.markdown(center_texte, unsafe_allow_html=True)
    st.write('')
    st.write('')
        #st.header(':green[**_Look at similar places in France_**]')



#########################
# Prediction Points sur la carte de France
#########################

    data = {
        'cellid': [5180546946359623680, 5189765251846897664, 5180949848651726848],
     'weight': [0.92, 0.74, 0.34]
     }

    df_test = pd.DataFrame(data,dtype='object')

    st_folium(bubble_plot(df_test), width = 825)




#########################
# images de lieux similaires en France
#########################

    # --> Insérer fonction plot_4pics_around de return.py

# Créer 3 colonnes d'égale largeur
    col1, col2, col3 = st.columns(3)

# Ajouter du contenu à la première colonne
    with col1:

        texte_1 = "Landscapes with 80% similarities"
        centrer_texte_1 = f"<center><h3>{texte_1}</h3></center>"

        st.markdown(centrer_texte_1, unsafe_allow_html=True)
        st.write('')


        img_test_path1 = 'app/images/bretagne.png'
        img_test1 = Image.open(img_test_path1)
        st.image(img_test1, caption='Your next french holidays', use_column_width=True)



# Ajouter du contenu à la deuxième colonne
    with col2:

        texte_2 = "Landscapes with 50% similarities"
        centrer_texte_2 = f"<center><h3>{texte_2}</h3></center>"

        st.markdown(centrer_texte_2, unsafe_allow_html=True)
        st.write('')

        img_test_path2 = 'app/images/bretagne.png'
        img_test2 = Image.open(img_test_path2)
        st.image(img_test2, caption='Your next french holidays', use_column_width=True)



# Ajouter du contenu à la 3ème colonne
    with col3:

        texte_3 = "Landscapes with 20% similarities"
        centrer_texte_3 = f"<center><h3>{texte_3}</h3></center>"

        st.markdown(centrer_texte_3, unsafe_allow_html=True)
        st.write('')

        img_test_path2 = 'app/images/bretagne.png'
        img_test2 = Image.open(img_test_path2)
        st.image(img_test2, caption='Your next french holidays', use_column_width=True)












############################
# Une fois l'API construite
###########################

    ### Get hexa from the upload file
    #image_hexa = Image.open(BytesIO(bytes.fromhex(uploaded_file)))


    # params = dict(img = uploaded_file,
    #               class_names = ?????)
    #travel_home_api_url = 'https://XXXXXXXXXX/predict'

    ### Make request to  API
    #response = requests.get(travel_home_api_url, params=params)


    #if response.status_code == 200:
        #### Display taggs returned by the API
        # prediction = response.json()['?????????']



        # st.header(f'')


    #else:
        # st.markdown("**Oops**, something went wrong 😓 Please try again.")
