from numpy import linspace
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from mpl_toolkits import mplot3d
from scipy import signal
 

import pandas as pd
import pathlib

# Extended requirements (might need to create a separate environment)
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Load data
ENCODER_HEAD_PATH = None
EMBEDDING_ROOT = pathlib.Path("embeddings")
LATEST_EMBEDDING = list(sorted(EMBEDDING_ROOT.glob("*")))[-1]
EMBEDDING_PATH = LATEST_EMBEDDING / "embeddings.pt"
EMBEDDING_PATH

# Code to create embeddings based on pretrained model
# Load embeddings
embeddings_raw = torch.load(EMBEDDING_PATH)

embedding_tensor = embeddings_raw["embeddings"]

for i in range(len(embedding_tensor)):
    print(embedding_tensor[i].shape)


if torch.cuda.is_available() and next(embedding_tensor.device()) == 'cuda':
  embedding_tensor = embedding_tensor.cpu()

flattened_embeddings = []
for tensor in embedding_tensor:
  # Assuming all tensors have compatible embedding dimension (128 in your case)
  flattened_embeddings.append(tensor.reshape(-1, tensor.shape[-1]))  # Flatten, move to CPU, convert to NumPy

# Concatenate the flattened NumPy arrays (assuming they are all NumPy arrays now)
embeddings = np.concatenate(flattened_embeddings, axis=0)

embeddings.shape

# First look into the data / find out about classes / label data

# Reduce dimensionality using PCA (adjust the number of components)
pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
embeddings_reduced = pca.fit_transform(embeddings[::6])  # Detach from gradients and convert to numpy array

# Prepare for plotting
# plt.figure(figsize=(8, 6))

# # Scatter plot the reduced embeddings
# for i, embedding in enumerate(embeddings_reduced[::50]):
#   plt.scatter(embedding[0], embedding[1])

# # Add labels and title
# plt.xlabel("Component 1")
# plt.ylabel("Component 2")
# plt.title("Visualization of Embeddings (PCA)")
# plt.legend()

# Show the plot
# plt.show()


# Look at explained variance
pca = PCA(n_components=embeddings.shape[1])
pca.fit(embeddings)
explained_variance = pca.explained_variance_ratio_
explained_variance_ratio = np.cumsum(explained_variance)


THRESHOLD = 0.90  

# Find the index where the cumulative explained variance ratio first exceeds the threshold
num_components = np.argmax(explained_variance_ratio >= THRESHOLD) + 1
num_components

pca = PCA(n_components=num_components)
pca_data = pca.fit_transform(embeddings)

# Extract the first 3 principal components
pca1 = pca_data[:, 0]
pca2 = pca_data[:, 1]
pca3 = pca_data[:, 2]

pca1

# Animation parameters
num_frames = 360  # Adjust for desired video length (more frames = smoother animation)
elevation = 0  # Adjust for starting elevation angle
angle = 0  # Initial rotation angle

# Creating 3D figure
# fig = plt.figure(figsize = (8, 8))
# ax = plt.axes(projection = '3d')
 
# # Creating Dataset
# ax.scatter3D(pca1[:100], pca2[:100], pca3[:100])
 
# # 360 Degree view
# for angle in range(0, 360):
#    ax.view_init(angle, 30)
#    plt.draw()
#    plt.pause(.001)
     
# plt.show()




# tSNE


perplexity = 30  # Adjust this based on your data and desired visualization
n_components = 3  # Number of dimensions to project to (2D for visualization)


for perplexity in [100]:
    # Apply t-SNE for dimensionality reduction
    tsne_model = TSNE(n_components=n_components, perplexity=perplexity)
    tsne_data = tsne_model.fit_transform(embeddings[::10])[:1000, :]

    fig = plt.figure(figsize = (8, 8))
    ax = plt.axes(projection = '3d')
    
    # Creating Dataset
    ax.scatter3D(tsne_data[:, 0],tsne_data[:, 1],tsne_data[:, 2])
    
    # 360 Degree view
    for angle in range(0, 360):
        ax.view_init(angle, 30)
        plt.draw()
        plt.pause(.001)
        
    plt.show()