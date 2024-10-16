
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
p_l = 4


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



Y= original_data #.detach().numpy()

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

Scoredata = output_before_att[:,sequence_size:]
Scoredata = Scoredata*100
latentdata = trajectories

alllevellatent = []
for s in range(dataset_size): 
    latent = []
    sample = (Scoredata[s, :])
    L = sample.copy()
    mask = sample.copy()
    MeanV = 0
    alllevel = []
    for i in range(p_l):
        # print (i)
        MeanV = np.median(L)
        sm = L <= MeanV
        lr = L > MeanV
        mask[mask <= MeanV] = i + 100
        L = L[lr]

   
    smellest = mask < 100
    mask[smellest] = i + 1 + 100
    dataset = pd.DataFrame({'data': s, 'rate': mask}, columns=['data', 'rate'])
    dataset2 = dataset.groupby(['rate']).mean()

    mask = mask.reshape(dataset_size, 1)
    unique, counts = np.unique(mask, return_counts=True)
    result = np.column_stack((unique, counts))

    sub = np.concatenate((latentdata, mask), axis=1)
    for j in [100, 101, 102, 103, 104]:
        sub2 = sub[:, sequence_size*latent_rep_size] == j  # Change based on tp

        sub3 = sub[sub2]
        sub3 = np.delete(sub3, -1, axis=1)
        sub3 = sub3.reshape(sub3.shape[0], sequence_size, latent_rep_size)  # shapeoftimeserise and latent
        sub3 = torch.from_numpy(sub3)
        sub3 = sub3.float()
        Avrglatent = torch.mean(sub3, dim=0)
        if (torch.isnan(Avrglatent).any()):
            print("True")
            print(s)
            print(j)
            exit()
        latent.append((Avrglatent))
    alllevellatent.append((latent))




Dpt = []



t =np.array([i for i in range (sequence_size)])
for q in range(dataset_size):
    record_id = q
    record_id = str(record_id)

    tt = t  
    tt = torch.from_numpy(tt)
    tt = tt.float()

    vals = original_data[q]
    mask = vals.copy()
    vals = torch.from_numpy(vals)
    vals = vals.float()

    where_are_NaNs = np.isnan(mask)
    mask[~where_are_NaNs] = 1
    mask[where_are_NaNs] = 0
 
    mask = torch.from_numpy(mask)

    mask = mask.float()

    labels = Y[q]
    labels = torch.from_numpy(labels)
    labels = labels.float()
 
    GODEL5 = alllevellatent[q][0]
    GODEL4 = alllevellatent[q][1]
    GODEL3 = alllevellatent[q][2]
    GODEL2 = alllevellatent[q][3]
    GODEL1 = alllevellatent[q][4]
    Dpt.append((record_id, tt, vals, mask , labels, GODEL5, GODEL4,GODEL3,GODEL2, GODEL1))

torch.save(
				Dpt,
				r'finaldata.pt')

