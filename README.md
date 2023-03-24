<h1 style="text-align:center;"> 🚂 🏡 TRAVEL HOME 🏠 🚌 </h1>

## Travel home is a 2-weeks Deep Learning project created by 4 students of Le Wagon Coding School (#117 batch).

<center>
<a href="https://www.youtube.com/watch?v=uXlWTxWLvlQ&ab_channel=LeWagon"> <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/YouTube_social_white_squircle.svg/2048px-YouTube_social_white_squircle.svg.png" width="100">
<a href="https://www.canva.com/design/DAFdKvY47Do/7HgkmPZFtftbXNEL98ltng/edit?utm_content=DAFdKvY47Do&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton"> <img src="https://medinsoft.com/wp-content/uploads/2020/01/lewagonlogocercle-1024x1024.png" width="100"></a>
</center>
<center>
<a href="https://nauvray-travel--travel-home-apptest-app-hortenseapp-test-45ia1q.streamlit.app/"> *** Try Me *** </a>
</center>

<h2 style="text-align: center;">Contributor</h2>

  ### <a href="https://github.com/hortense-jallot"> [Hortense Jallot] </a> - Front end & Model
  ### <a href="https://github.com/mle64"> [Maddalen Lepphaille] </a> - Preprocess & API
  ### <a href="https://github.com/Marie-Pierre74"> [Marie-Pierre Jacquemin] </a> - Geohash and Front end
  ### <a href="https://github.com/nauvray"> [Nicolas Auvray] </a> - GCP & Docker

&nbsp;

If you have any question regarding the project feel free to ask any of the contributor. Even if we had to specialize during this project we all have a good overview of the project.

&nbsp;

<h2 style="text-align: center;">Description</h2>

  Travel-home is an interface which provides you a selection of locations at a train/car distance similar to your dream travel destination.
  Concretely, from an image that you upload or simply a keyword, the application displays a map of France with the 3 destinations that are the closest to the original photo.

<center>
<img src=https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1869px-Python-logo-notext.svg.png alt="Python" height="100">
<img src=https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/PyTorch_logo_icon.svg/640px-PyTorch_logo_icon.svg.png alt="Pytorch" height="100">
<img src=https://docs.s2cell.aliddell.com/en/stable/_static/logo.min.svg alt="S2 Geometry" height="100">
<img src=https://www.docker.com/wp-content/uploads/2022/03/vertical-logo-monochromatic.png alt="Docker" height="100">
<img src=https://miro.medium.com/v2/resize:fit:256/1*CHzvR53_W9FR2s1BQW9Bqg.png alt="Google Cloud Storage" height="100">
<img src=https://cloudacademy.com/wp-content/uploads/2014/04/ComputeEngine_512px-300x300.png alt="Google Cloud Compute" height="100">
<img src=https://carlossanchez.files.wordpress.com/2019/06/21046548.png alt="Google Cloud Registry" height="100">
<img src=https://res.cloudinary.com/crunchbase-production/image/upload/c_lpad,f_auto,q_auto:eco,dpr_1/z3ahdkytzwi1jxlpazje alt="Streamlit" height="100">
<img src=https://cdn.worldvectorlogo.com/logos/fastapi.svg alt="Fast API" height="100">
<img src=https://tarun-kamboj.github.io/images/tools/folium.png alt="Folium" height="100">
<img src=https://numfocus.org/wp-content/uploads/2016/07/pandas-logo-300.png alt="Pandas" height="100">
<img src=https://seeklogo.com/images/N/numpy-logo-479C24EC79-seeklogo.com.png alt="Numpy" height="100">
<img src=https://upload.wikimedia.org/wikipedia/commons/thumb/0/01/Created_with_Matplotlib-logo.svg/1024px-Created_with_Matplotlib-logo.svg.png alt="Matplotlib" height="100">
<img src=https://seaborn.pydata.org/_images/logo-tall-lightbg.svg alt="Seaborn" height="100">

</center>

&nbsp;

<h2 style="text-align: center;">The Project</h2>

<h3>Initial idea</h3>
France is full or various landscape, richer than we could expect. "Region Bretagne" and its ad campaign played on this aspect and inspired us.
<center>
<img src=https://www.tourisme-rennes.com/uploads/2020/06/Caraibzh-Bretagne.jpg >
</center>

<details>
<summary><font size="+1">Dataset</font></summary>
4.2 millions pictures geotagged around the world and 660K geotagged around France (~130Go of data). All pictures are gathered in a zipfile containing 142 msgpack folder with each picture and its metadata.

>&rarr;<ins>**Output**</ins>  is 142 csv file of each pictures taken close to France with :
  >- name
  >- latitude
  >- longitude
  >- data (the picture) in hex
<center>
<a href="https://www.kaggle.com/datasets/habedi/large-dataset-of-geotagged-images"> <img src="https://upload.wikimedia.org/wikipedia/commons/7/7c/Kaggle_logo.png" height=100> </a>
</center>
</details>

<details>
<summary><font size="+1">Geohashing and Geotagging</font></summary> Segmentation of France into 1070 squares of around 300 pictures. The goal is to associate each photo to their corresponding square.

  >&rarr; <ins>**Output**</ins> is the addition of a column in the 142 csv with the square name corresponding to the lattitude and longitude where the picture has been taken.
</details>

<details>
<summary><font size="+1">Pre-processing</font></summary>Using the original ResNet18 to remove pictures not showing any relevant scene attribute. Then all pictures contained in a square will be gathered in a folder.

  >&rarr; <ins>**Output**</ins> are all the pictures showing landscape attributes. They are all splitted in a folder of the name of their corresponding square.
</details>

<details>
<summary><font size="+1">Model</font></summary>We used the model Resnet18 pre-trained on the dataset Places365, and adaptated it to our usecase. To summarize we trained our model not to predict labels but cellid (with similar scene attributes) as the inpute picture.

  >&rarr; <ins>**Output**</ins> are 3 locations (squares) with similar scene attributes as the input picture
</details>

<details>
<summary><font size="+1">API</font></summary> We implemented an api which returns the prediction of the model : <a href="https://travel-home-mzfiw6j4fa-ew.a.run.app/docs#"> API Link </a>

  >&rarr; <ins>**Output**</ins> is an online API returning the 3 most similar squares.
</details>

<details>
<summary><font size="+1">Interface</font></summary> We deployed an interface on Streamlit : <a href="https://nauvray-travel--travel-home-apptest-app-hortenseapp-test-45ia1q.streamlit.app/"> Website here</a>

  >&rarr; <ins>**Output**</ins> is an online interface with as input : a search engine and an area for uploading picture. As output it shows the location of the 3 prefered squares and some similar picture it contains.
</details>

&nbsp;

<h2 style="text-align: center;">Sources</h2>

<h3>Bibliography</h3>

<em>[1] Eric Müller-Budack, Kader Pustu-Iren and Ralph Ewerth : Geolocation Estimation of Photos using a Hierarchical Model and Scene Classification.</em>

<em>[2] Weyand, T., Kostrikov, I., Philbin, J.: Planet-photo geolocation with convolutional neural networks. In: European Conference on Computer Vision. pp. 37–55. Springer (2016).</em>

<h3>Online</h3>

<a href="http://places2.csail.mit.edu/"> Places2 </a> - Places dataset has been used to train various model from the VGG16 to ResNet150. This website is the MIT official website of the group of researcher who set up this dataset. It contains also a demo of one model.

<a href="https://github.com/CSAILVision/places365"> GitHub - Places2</a> - Github of all the models developped and using the Places365 dataset previously mentionned.

<a href="https://s2geometry.io/"> S2 Geometry - Google </a> - S2 is the program initially set up (and dropped) by Google. Their job has been to split world in squares (called cellid) and provide them a normated name. Code has been done in C++ and need to be compiled (with CMake) to be used in Python.

<a href="https://docs.s2cell.aliddell.com/en/stable/"> s2cell </a> - This is the Python library which takes the same mathematical principals as the original S2 geometry.
