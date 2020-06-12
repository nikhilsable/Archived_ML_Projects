import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"
from sklearn.cluster import KMeans
import seaborn as sns

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

# Common imports
import numpy as np
import os
import pandas as pd

# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "rnn"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


df = pd.read_csv('data_sources/raw_pp_seed_data_9739_outlier_analysis.csv', index_col='time')
df.index = pd.to_datetime(df.index.values)

#Reindex to identify missing values
#df = df_reindexer(df, timesteps='H')

#handle missing values
df = df.ffill()

df.columns = ['peakpower_on', 'pk2pk', 'prelase', 'pzt', 'pk2pk_3sig', 'peak_power_off_3sig', 'pk2pk_oos', 'peakpower_off']

#context filtering
uel = 650
lel = 100

df = df[(df.peakpower_off <= uel) & (df.peakpower_off >= lel)]

df.prelase = df.prelase * 1000000 #converting microseconds to seconds

df.pk2pk_3sig = df[df.pk2pk_3sig <= 25]

df = df.dropna()

#intro plots
sns.pairplot(df)

#Setup training data
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

k = 2
kmeans = KMeans(n_clusters=k, random_state=42)
y_pred = kmeans.fit_predict(X)

#cluster centers
kmeans.cluster_centers_

#index of the cluster that instance gets assigned to
kmeans.labels_

df = df.reset_index()
df['cluster'] = pd.Series(y_pred)
df = df.set_index('index')

sns.pairplot(df, hue='cluster')

df.cluster.plot(style='8', figsize=(18,13))
plt.show()

df.pk2pk_oos.plot(style='8', figsize=(18,13))
plt.show()

df.pzt.plot(style='8', figsize=(18,13))
plt.show()

sns.pairplot(df[['cluster', 'pzt']], hue='cluster', ).figsize=(18,13)
