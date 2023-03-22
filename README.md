# TRAVEL_HOME
Travel home is a 2-weeks Deep Learning project created by 4 students of Le Wagon Coding School (#117 batch).
It is an interface which provides you a selection of locations at a train/car distance similar to your dream travel destination. 
Concretely, from an image that you upload or simply a keyword, the application displays a map of France with the 3 destinations that are most close to the original photo.

**Dataset**: 660K pictures geotagged around France ~130Go of data.

**Geohashing and Geotagging**: Segmentation of France into 4053 squares of around 100 pictures.

**Pre-processing**: Filtering of the Dataset : -> 440K pictures and associate each picture to a geotag (cellid).

**Model**: We used the model Resnet18 pre-trained on the dataset Places365, and adaptated it to our usecase.

**Api**: We implemented an api which returns the prediction of the model : https://travel-home-mzfiw6j4fa-ew.a.run.app/docs#

**Interface**: We deployed an interface on Streamlit : https://nauvray-travel--travel-home-apptest-app-hortenseapp-test-45ia1q.streamlit.app/

## Setup

Run this command in your terminal to setup the model :

```bash
curl https://raw.githubusercontent.com/nauvray/travel-home 
```


All the directories and modules will be set up for the adapted model.
