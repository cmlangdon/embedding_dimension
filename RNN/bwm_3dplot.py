import os

import matplotlib.pyplot as plt
import matplotlib
plt.rcParams["axes.grid"] = False
matplotlib.rcParams['axes.linewidth'] = .5
import pandas as pd

import matplotlib.pyplot as plt

import os





# Load results
df_list = []
for filename in os.listdir('/Users/cl1704/PycharmProjects/embedding_dimension/BWM/Results'):
    f = os.path.join('/Users/cl1704/PycharmProjects/embedding_dimension/BWM/Results', filename)
    if os.path.isfile(f):
        df_list.append(pd.read_pickle(f))
df = pd.concat(df_list,ignore_index=True)

df= df[df.n_neurons>=50]
#df = df[df.pca_1>.5]
#df = df[df.linear_error_test<1]


import plotly.express as px

fig = px.scatter_3d(df, x='kmeans_error', y='linear_error_train_mean', z='pca_error')

fig.show()