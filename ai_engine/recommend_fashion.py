import os
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold import TSNE
import tensorflow as tf
import keras
from keras import Model
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import GlobalMaxPooling2D

# Set dataset path
DATASET_PATH = "D:/Python/personal stylish/data/fashion-dataset/"

# Load dataset
df = pd.read_csv(DATASET_PATH + "styles.csv", nrows=5000)
df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)
df = df.reset_index(drop=True)

def plot_figures(figures, nrows=1, ncols=1, figsize=(8, 8)):
    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
    for ind, title in enumerate(figures):
        axeslist.ravel()[ind].imshow(cv2.cvtColor(figures[title], cv2.COLOR_BGR2RGB))
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout()

def img_path(img):
    return os.path.join(DATASET_PATH, "images", img)

def load_image(img, resized_fac=0.1):
    img = cv2.imread(img_path(img))
    w, h, _ = img.shape
    resized = cv2.resize(img, (int(h * resized_fac), int(w * resized_fac)), interpolation=cv2.INTER_AREA)
    return resized

# Load and display sample images
figures = {'im' + str(i): load_image(row.image) for i, row in df.sample(6).iterrows()}
plot_figures(figures, 2, 3)

df.articleType.value_counts().sort_values().plot(kind='barh')
plt.show()

# Define ResNet50 model
img_width, img_height, _ = 224, 224, 3
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
base_model.trainable = False

model = keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

def get_embedding(model, img_name):
    img = image.load_img(img_path(img_name), target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x).reshape(-1)

# Compute embeddings
df_sample = df
map_embeddings = df_sample['image'].apply(lambda img: get_embedding(model, img))
df_embs = pd.DataFrame(map_embeddings.tolist())

# Compute cosine similarity matrix
cosine_sim = 1 - pairwise_distances(df_embs, metric='cosine')

def get_recommender(idx, df, top_n=5):
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    idx_rec = [i[0] for i in sim_scores]
    return df.iloc[idx_rec].index, [i[1] for i in sim_scores]

# Example recommendations
idx_ref = 2993
idx_rec, idx_sim = get_recommender(idx_ref, df, top_n=6)
plt.imshow(cv2.cvtColor(load_image(df.iloc[idx_ref].image), cv2.COLOR_BGR2RGB))
figures = {'im' + str(i): load_image(row.image) for i, row in df.loc[idx_rec].iterrows()}
plot_figures(figures, 2, 3)

# t-SNE visualization
time_start = time.time()
tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(df_embs)
df['tsne-2d-one'], df['tsne-2d-two'] = tsne_results[:, 0], tsne_results[:, 1]
print(f't-SNE done! Time elapsed: {time.time()-time_start} seconds')

plt.figure(figsize=(16, 10))
sns.scatterplot(x="tsne-2d-one", y="tsne-2d-two", hue="masterCategory", data=df, legend="full", alpha=0.8)
plt.show()

plt.figure(figsize=(16, 10))
sns.scatterplot(x="tsne-2d-one", y="tsne-2d-two", hue="subCategory", data=df, legend="full", alpha=0.8)
plt.show()

# Save outputs
df.sample(10).to_csv('df_sample.csv')
df_embs.to_csv('embeddings.csv')
df.to_csv('metadados.csv')
