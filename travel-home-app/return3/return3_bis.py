import pandas as pd
import s2cell
import matplotlib.pyplot as plt
import numpy as np




########################################################################
### NOUVELLE FONCTION
########################################################################


def plot_4pics_around(df_predict):
    for cellid in df_predict.cellid:
