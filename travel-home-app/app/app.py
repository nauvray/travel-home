import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
from return3.return3 import bubble_plot
from streamlit_folium import st_folium
import os
from st_clickable_images import clickable_images

# commande terminal =  streamlit run app/app.py

##################################################
############################ Fonctions de Nico
##################################################



def launch_plexel(word:str):
    photo = 'https://api.unsplash.com/search/photos'
    my_params = {'query':word,'client_id':'7ANOoawIlsbj-XMwK6am_kjkYwN_w-TnsNfgz0aKHFU'}
    x = requests.get(photo,params=my_params)
    x.json()
    list_link=[]
    for i in range(len(x.json()['results'])):
        if i < 5 and x.json()['results'][i]['urls']['small'] not in list_link:
            list_link.append(x.json()['results'][i]['urls']['small'])

    return list_link

##################################################
############################ END Fonctions de Nico
##################################################


st.set_page_config(layout="wide")

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


#########################
# insertion logo
#########################

logo = "app/images/LogoGoogleSust1.png"

# Créer un conteneur
with st.container():
# Ajouter le logo dans le conteneur
    st.image(logo, width=60)



##############################
# Créa de 2 lignes de téléchargement texte ou image
##############################


# Avec la barre de saisie (texte)
################################
st.subheader('_**Write your dream destination and press Enter**_')
selected = st.text_input("")

image_uploaded = None

if selected:
    list_link=launch_plexel(selected)

    text = 'Now, select your favorite image.... and Wait ....'
    center_text = f"<center><h3>{text}</h3></center>"
    st.markdown(center_text, unsafe_allow_html=True)

    clicked = clickable_images(
    list_link,
    titles=[f"Image #{str(i)}" for i in range(5)],
    div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
    img_style={"margin": "5px", "height": "200px"},
    )
    if clicked > -1:
        url = list_link[clicked]
        response = requests.get(url)
        image_uploaded = Image.open(BytesIO(response.content))
        col1, col2, col3 = st.columns([3,3,3])
        with col1:
            st.write('')
        with col2:
            st.image(image_uploaded, caption='Your Dream Destination', use_column_width=True)
        with col3:
            st.write('')



else:
    st.write('')
    st.subheader('_**Or upload here your own picture**_')
    uploaded_file = st.file_uploader('', type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        image_uploaded = Image.open(uploaded_file)

        # Affiche l'image redimensionnée
        col1, col2, col3 = st.columns([3,3,3])
        with col1:
            st.write('')
        with col2:
            st.image(image_uploaded, caption='Your Dream Destination', use_column_width=True)
        with col3:
            st.write('')



#########################
# Prediction Points sur la carte de France + Photos du même square/cellid
#########################



if image_uploaded:
## --> Texte d'accroche
    texte = "Discover now your next holidays spots in France"
    center_texte = f"<center><h2>{texte}</h2></center>"

    st.markdown(center_texte, unsafe_allow_html=True)
    st.write('')
    st.write('')


## --> Intégration des données de l'API

############################
# Une fois l'API construite
###########################


    # params = dict(img = image_uploaded)
    #travel_home_api_url = 'https://XXXXXXXXXX/predict'

    ### Make request to  API
    #result = requests.get(travel_home_api_url, params=params)


    #if result.status_code == 200:
        #### Display taggs returned by the API
        # prediction = result.json()['image']


    #else:
        # st.markdown("**Oops**, something went wrong 😓 Please try again.")


## --> Transformation du dictionnaire (pred du modele)

## si dictionnaire des predicts de Hortense se nomme prediction (avec une colonne 'probability' et une colonne 'sell_id')
# construction du DataFrame relatif pour appliquer la fonction bubble_plot
# df_predict = pd.DataFrame(prediction).rename(columns={'probability':'weight', 'sell_id' :'cellid'})


## --> Affichage de la carte de france

    col1, col2, col3 = st.columns([2,6,1])

    with col1:
        st.write("")

    with col2:
        data = {
                'cellid': [5180546946359623680, 5189765251846897664, 5180949848651726848],
                'weight': [0.92, 0.74, 0.34]
                }

        df_test = pd.DataFrame(data,dtype='object')
        st_folium(bubble_plot(df_test),width = 825)

    with col3:
        st.write("")




## --> images de lieux similaires en France


    # --> Insérer fonction qui prend 4 images au hasard d'un même cellid et les plotter (A Faire avec Maddalen)

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
