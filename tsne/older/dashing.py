from dash import Dash, dcc, html, Input, Output, no_update, callback
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os

reduced_vectors = np.load('tsne-out.npy')

# Path to the folder containing images
image_folder = 'assets/outputs'

# Get the list of image filenames
image_files = [f for f in os.listdir(image_folder) if f.endswith(('png', 'jpg', 'jpeg', 'gif'))]

# Create the full image paths
image_paths = [os.path.join(image_folder, img) for img in image_files]

df = pd.DataFrame({
    'x': reduced_vectors[:, 0],
    'y': reduced_vectors[:, 1],
    'image': image_paths  # Store the image paths in a column
})

fig = go.Figure(data=[
    go.Scatter(
        x=df['x'],
        y=df['y'],
        mode="markers",
    )
])

# turn off native plotly.js hover effects - make sure to use
# hoverinfo="none" rather than "skip" which also halts events.
fig.update_traces(hoverinfo="none", hovertemplate=None)

fig.update_layout(
    xaxis=dict(title='x'),
    yaxis=dict(title='y'),
    plot_bgcolor='rgba(255,255,255,0.1)'
)

app = Dash()

app.layout = html.Div([
    dcc.Graph(id="graph-basic-2", figure=fig, clear_on_unhover=True),
    dcc.Tooltip(id="graph-tooltip"),
])


@callback(
    Output("graph-tooltip", "show"),
    Output("graph-tooltip", "bbox"),
    Output("graph-tooltip", "children"),
    Input("graph-basic-2", "hoverData"),
)
def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update

    # demo only shows the first point, but other points may also be available
    pt = hoverData["points"][0]
    bbox = pt["bbox"]
    num = pt["pointNumber"]

    df_row = df.iloc[num]
    img_src = df_row['image']

    children = [
        html.Div([
            html.Img(src=img_src, style={"width": "100%"}),
        ], style={'width': '400px', 'white-space': 'normal'})
    ]

    return True, bbox, children


if __name__ == "__main__":
    app.run(debug=True)