
"""
###########################
# SeqLink: A Robust Neural-ODE Architecture for Modelling Partially Observed Time Seriesme Series
# Author: Futoon Abushaqra
###########################



"""

from tensorflow.keras.layers import concatenate
from keras_self_attention import SeqSelfAttention
import keras
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten,  MaxPooling1D, Conv1D, Conv2D,Embedding, Reshape, Attention
from pandas import read_csv
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from math import sqrt
from matplotlib import pyplot
from keras.layers import LSTM
from pandas import datetime
from sklearn.metrics import mean_squared_error, r2_score
from numpy import array
from attention import Attention
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
from keras.models import model_from_json
import matplotlib.pylab as plt

dataset_size = 100  # 500
sequence_size = 100
dim_size = 1
latent_rep_size = 10



trajectories = read_csv(r'dataRep\D24.csv' , header=None)
trajectories = trajectories.values

original_data = read_csv(r'E:\SeqLink24\data\PeriodocData\10010.csv' , header=None)
original_data.fillna(0, inplace=True)
original_data = original_data.values
original_data = original_data.shape
x1 = original_data.reshape(dataset_size,1,sequence_size) #univariate

#x1 = actual_data.reshape(datasetsize,sequencessieze, dimsizes) #multivariate  data

X2 = np.empty([datasetsize,datasetsize,(sequencessieze*latentRepsize)])

for i in range(dataset_size):
    for j in range (dataset_size):
        X2[i][j] = Tr[j]



Y= data #.detach().numpy()

inp1 = keras.Input(shape = (dim_size, sequence_size))
inp2 = keras.Input(shape = (dataset_size, sequence_size*latent_rep_size))
inp1Em = Dense(32, activation="relu")(inp1)
inp2Em = Dense(32, activation="relu")(inp2)
Combine = concatenate([inp1Em,inp2Em], axis=1)
M = Attention()(Combine)
M = Dense(100, activation="relu") (M)
mod = keras.Model(inputs=(inp1,inp2), outputs=M)
mod.compile(optimizer="adam", loss="mse")
mod.fit(
  	x=[x1,X2], y=Y,
   	epochs=100, batch_size=50)


new_model = keras.Model(inputs=mod.input, outputs=mod.get_layer(name="attention_weight").output)
output_before_att = new_model.predict([x1,X2])
#ax = sns.heatmap(output_before_att, linewidth=0.5)
#plt.show()

np.savetxt('Attention_weights.csv',output_before_att, delimiter=",")

