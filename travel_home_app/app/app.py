import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
from streamlit_folium import st_folium
import os
from st_clickable_images import clickable_images
import s2cell
import s2sphere
from s2sphere import CellId, LatLng, Cell
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import geopy
from geopy.geocoders import Nominatim
from io import BytesIO
import subprocess
import random
from travel_home.params import *


# commande terminal =  streamlit run app/app.py

########################################################################
##### fonction qui à partir d'un DataFrame (avec une col cellid et une col weight) retourne le centre
########################################################################

def bubble_plot(df_result):
########################################################################
### PART 1
########################################################################
    # create new dataframe with center and % of weight
    df_result[['lat','lon']] = df_result.apply(lambda x: s2cell.cell_id_to_lat_lon(x.cellid), axis=1, result_type='expand')

    # create a new column = weight in %
    df_result['new_weight'] = df_result['probability'].apply(lambda x: round(x*100))


########################################################################
### PART 2
########################################################################
    threshold = 10

    # create a new dataframe with only % > threshold

    df_select = df_result[df_result.new_weight > threshold]


########################################################################
### PART 3
########################################################################
    # plot the bubble around the center + weight
    #### Avec FOLIUM


    geolocator = Nominatim(user_agent="my-app")

    location = geolocator.geocode("France")

    map_fr = folium.Map(location=[location.latitude, location.longitude], zoom_start=6)



    for lat, lon, weight in zip(df_select['lat'], df_select['lon'], df_select['new_weight']):
        folium.CircleMarker(location=[lat, lon],
                  radius=weight/2,
                  color='blue',
                  fill=True,
                  fill_color='blue').add_to(map_fr)


    return map_fr

########################################################################
##### END Fonction Bubbleplot
########################################################################


########################################################################
### NOUVELLE FONCTION pour plotter les 4 images par cellid predict
########################################################################


def plot_4pics_around(cellid):
    my_local_path = '/Users/marie/code/Marie-Pierre74/travel-home/00-data/img_test'
    cellid_path =  f'gs://{BUCKET_NAME}/npy/{cellid}'
    subprocess.call(['gsutil', '-m', 'cp', '-r', cellid_path, my_local_path])

    image_path = os.path.join(my_local_path,cellid)

    nb_images = 4
    count = 1

    for i in range(nb_images):

        file_name = random.choice(os.listdir(image_path))
        img_array = load_npy_image(image_path,file_name)
        plt.subplot(nb_images,1,count)
        plt.imshow(img_array)

        #Remove plot ticks
        plt.xticks(())
        plt.yticks(())
        count +=1
    return plt.gcf()


#######################################################################
### END NOUVELLE FONCTION pour plotter les 4 images par cellid predict
########################################################################


##################################################
############################ Fonction Maddalen de utils
##################################################


def load_npy_image(npy_path : str, npy_file : str) -> np.ndarray:
    '''load image (numpy.array) from npy file'''
    npy_file_path = os.path.join(npy_path, npy_file)
    img_array = np.load(npy_file_path)
    print(f"npy image loaded with shape: ({img_array.shape[0]}, {img_array.shape[1]}, 3)")
    return img_array

##################################################
############################ END Fonction Maddalen de utils
##################################################


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


# Avec la barre d'upload (image)
################################

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
# Prediction : Bubbles sur la carte de France + Photos du même square/cellid
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


    #params = dict(img = image_uploaded)
    #url = 'https://XXXXXXXXXX/predict'

    ### Make request to  API
    #result = requests.get(url, params=params)


    #if result.status_code == 200:
        #### Display taggs returned by the API
        # prediction = result.json()['img']


    #else:
        # st.markdown("**Oops**, something went wrong 😓 Please try again.")


## --> Transformation du dictionnaire (pred du modele)

## si dictionnaire des predicts de Hortense se nomme prediction (avec une clé 'probability' et une clé 'cellid')

# !!!!!!!!!!!!!!!  Décommenter la 2ème ligne ci-dessous pour activer le DataFrame df_predict

# construction du DataFrame relatif pour appliquer la fonction bubble_plot
# df_predict = pd.DataFrame(prediction,dtype='object')


## --> Affichage de la carte de france

    col1, col2, col3 = st.columns([2,6,1])

    with col1:
        st.write("")

    with col2:
        prediction = {
                'cellid': [1343598811095760896, 5169868489430663168, 5218837438796922880],
                'probability': [0.83, 0.56, 0.34]
                }

        df_predict = pd.DataFrame(prediction,dtype='object')
        # ici prendre le df_predict recréé au-dessus après l'avoir activé (décommenté)
        st_folium(bubble_plot(df_predict),width = 825)

    with col3:
        st.write("")




## --> images de lieux similaires en France


    # --> Insérer fonction qui prend 4 images au hasard d'un même cellid et les plotter (A Faire avec Maddalen)

# Créer 3 colonnes d'égale largeur
    col1, col2, col3 = st.columns(3)

    # Ajouter du contenu à la première colonne  !!!!!!!! changer le DataFrame en prenant df_predict à la place de df_test
    with col1:
        weight_1 = df_predict['probability'][0]
        texte_1 = f"Landscapes with {round(weight_1*100)}% similarities"
        centrer_texte_1 = f"<center><h3>{texte_1}</h3></center>"

        st.markdown(centrer_texte_1, unsafe_allow_html=True)
        st.write('')

        fig_1 = plot_4pics_around(str(df_predict['cellid'][0]))
        st.pyplot(fig_1)

    # Ajouter du contenu à la deuxième colonne  !!!!!!!! changer le DataFrame en prenant df_predict à la place de df_test
    with col2:
        weight_2 = df_predict['probability'][1]
        texte_2 = f'Landscapes with {round(weight_2*100)}% similarities'
        centrer_texte_2 = f"<center><h3>{texte_2}</h3></center>"

        st.markdown(centrer_texte_2, unsafe_allow_html=True)
        st.write('')

        fig_2 = plot_4pics_around(str(df_predict['cellid'][1]))
        st.pyplot(fig_2)

    # Ajouter du contenu à la 3ème colonne  !!!!!!!! changer le DataFrame en prenant df_predict à la place de df_test
    with col3:
        weight_3 = df_predict['probability'][2]
        texte_3 = f'Landscapes with {round(weight_3*100)}% similarities'
        centrer_texte_3 = f"<center><h3>{texte_3}</h3></center>"

        st.markdown(centrer_texte_3, unsafe_allow_html=True)
        st.write('')

        fig_3 = plot_4pics_around(str(df_predict['cellid'][2]))
        st.pyplot(fig_3)

###############
######END
##############
