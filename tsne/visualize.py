from dash import Dash, dcc, html, Input, Output, no_update, callback
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os

TSNE_FILE = 'data/tnse.csv'
IMAGE_PATH = 'assets/outputs/'

df = pd.read_csv(TSNE_FILE)
df['image'] = df['font'].apply(lambda x: f"{IMAGE_PATH}{x}{'.png'}")
print(df['image'].iloc[4])

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
    plot_bgcolor='rgba(255,255,255,0.1)',
    title="t-SNE Visualization of FontSpace Data",
    title_x=0.5,
    title_y=0.95,
    title_font=dict(size=30, color="black", family="Arial"),
    height=800
)

app = Dash()

app.layout = html.Div([
    dcc.Graph(id="graph-basic-2", figure=fig, clear_on_unhover=True),
    dcc.Tooltip(id="graph-tooltip"),
], style={
    'height': '100%'
})

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
    app.run(debug=True, port=18812, host='0.0.0.0')
