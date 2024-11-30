import plotly.express as px
import numpy as np
import os
import pandas as pd

# Load the t-SNE data
reduced_vectors = np.load('tsne-out.npy')

# Path to the folder containing images
image_folder = '/mnt/storage/smagid/thesis-smagid/imagedisp/outputs'

# Get the list of image filenames
image_files = [f for f in os.listdir(image_folder) if f.endswith(('png', 'jpg', 'jpeg', 'gif'))]

# Create the full image paths
image_paths = [os.path.join(image_folder, img) for img in image_files]

# Create a DataFrame with x, y, and custom image paths
df = pd.DataFrame({
    'x': reduced_vectors[:, 0],
    'y': reduced_vectors[:, 1],
    'image': image_paths  # Store the image paths in a column
})

# Create the scatter plot using the DataFrame
fig = px.scatter(
    df,
    x='x',
    y='y',
    title="TypeFace Space T-SNE Map"
)

fig.update_traces(
    hovertemplate='<b>X: %{x}</b><br><b>Y: %{y}</b><br><img src="%{customdata}" width="100" height="100">',
    customdata=df['image'],
    hoverinfo="text"
)

# Show the plot
fig.show()