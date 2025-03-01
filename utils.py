# code to display the variables of the dataset (don't pay too much attention)

from sys import exec_prefix
from IPython.display import display, HTML

def display_variables(df):
    key_labels = {
        "Si" : "SizeDistrLogNormBin",
        "SS" : "SSA",
        "Ro" : 'Ross Parameters',
        "Soluble_Fraction_" : "Soluble Fraction",
        "Soluble_Volume_Concentration" : "Soluble Volume Concentration",
        "Water_Fraction" : "Water Fraction",
        "Water_Volume_Concentration" : "Water Volume Concentration",
        "Land" : "Land parameters",
        "RealRefInd" : "Reflection indices (?)",
        "Cox_Munk" : "Cox Munk Parameters",
        "DHR" : "DHR Parameters",
        "BrC" : "BrC volume concentration / fraction",
        "Insoluble_Fraction" : "Insoluble Fraction",
        "Insoluble_Volume_Concentration" : "nsoluble Volume Concentration",
        "Iron" : "Iron volume concentration / fraction",

    }

    grouped_strings = {}

    for key in key_labels:
        for string in df.columns:
            if string[:len(key)] == key:
                if key in grouped_strings:
                    grouped_strings[key].append(string)
                else:
                    grouped_strings[key] = [string]


    html_output = "<h2>Variables</h2>"
    html_output += "<table>"
    html_output += "<tr><th>Category</th><th>Variables</th></tr>"

    for key, strings in grouped_strings.items():
        category_label = key_labels.get(key, 'Other')
        try:
            strings_html = ", ".join(strings)
            html_output += f"<tr><td>{category_label}</td><td>{strings_html}<br/></td></tr>"
        except:
            pass

    html_output += "</table>"

    html_output += f"<h2>RAM Usage : {(df.memory_usage().sum() / 1024**2):.2f} MB</h2>"

    # Display HTML
    display(HTML(html_output))

from plotly import graph_objects as go
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

mapbox_access_token = "pk.eyJ1IjoibWF0aGlzdzU5IiwiYSI6ImNsaDZsYWs2czA3YWkzZnBlMnhtcmhyYW4ifQ.imLZJq1w2W6-yhuPQEb16Q"

import networkx as nx


def reorder_clusters(cluster_centers, labels):
    """ Reorder the clusters using the traveling salesman problem """
    distance_matrix = (cluster_centers[np.newaxis, :, :] - cluster_centers[:, np.newaxis, :])**2
    distance_matrix = np.sqrt(np.sum(distance_matrix, axis=-1))

    num_points = len(cluster_centers)

    G = nx.complete_graph(num_points)

    # Assign distances as edge weights
    for i in range(num_points):
        for j in range(num_points):
            if i != j:
                G[i][j]['weight'] = distance_matrix[i][j]

    tsp_solution = np.array(nx.approximation.traveling_salesman_problem(G, cycle=False))

    tsp_edges = [(tsp_solution[i], tsp_solution[i+1]) for i in range(len(tsp_solution)-1)]
    tsp_edges.append((tsp_solution[-1], tsp_solution[0]))  # Add edge from last node to first node to complete the cycle

    clusters_center = cluster_centers.mean(axis=0)
    farthest_cluster = np.argmax(np.linalg.norm(cluster_centers - clusters_center, axis=1))

    start_idx = np.where(tsp_solution == farthest_cluster)[0][0]
    tsp_solution = np.roll(tsp_solution, -start_idx)

    new_labels = np.zeros_like(labels)
    for i, cluster in enumerate(tsp_solution):
        new_labels[labels == cluster] = i

    return new_labels


def plot_variable(df, variable, colormap = cm.viridis):

    cmap = lambda x : f'rgba{tuple(list((np.array(colormap(x)) * 255)[:3].astype(int)) + [128])}' # function to convert the values to corresponding rgba code

    mask = np.logical_not(np.isnan(df[variable])) # we only want to plot the non nan values

    norm = plt.Normalize(min(df[variable][mask]), max(df[variable][mask])) # we normalize the values to get the colors
    normalized_data_values = norm(df[variable][mask])

    viridis_colors = [cmap(x) for x in normalized_data_values] # we get the colors

    trace = go.Scattermapbox(   # we create the scattermapbox plot
        lat=df['Latitude'][mask],
        lon=df['Longitude'][mask],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=10,
            color=viridis_colors
        ),
        text=df[variable][mask].map("Value : {:.3f}".format)
    )

    fig = go.Figure(data=trace) # we create the figure

    fig.update_layout( # we update the layout
        autosize=True,
        hovermode='closest',
        mapbox=dict(
            center=dict(
                lat=0,
                lon=0
            ),
            accesstoken=mapbox_access_token,
            bearing=0,
            pitch=0,
            zoom=1,
            style="satellite"
        ),
        width = 1500,
        height = 750
    )

    fig.show()

def plot_clusters(df, variable, cluster_centers, colormap = cm.viridis):
    labels = df[variable]
    df[variable] = reorder_clusters(cluster_centers, labels)
    plot_variable(df, variable, colormap)

from matplotlib.colors import ListedColormap
import numpy.ma as ma

def discrete_matshow(data):
    # get discrete colormap
    cmap = plt.get_cmap('RdBu', np.max(data) - np.min(data) + 1)
    # set limits .5 outside true range
    mat = plt.matshow(data, cmap=cmap, vmin=np.min(data) - 0.5,
                      vmax=np.max(data) + 0.5)
    # tell the colorbar to tick at integers
    cax = plt.colorbar(mat, ticks=np.arange(np.min(data), np.max(data) + 1))

def show_results(labels, mask, img_shape = (180, 360)):
  # Reshape labels back to the original shape of the variables
  label_image = labels.reshape(img_shape)  # Assuming all variables have the same shape
  plt.figure(figsize=(16,10))
  # Plot the clustering result
  # Assuming label_image is your label array
  unique_labels = np.unique(label_image)
  n_labels = len(unique_labels)

  # Create a custom colormap with n_labels colors
  # Here, we're using a part of the 'Set1' colormap
  custom_cmap = ListedColormap(plt.cm.tab20(np.linspace(0, 1, n_labels)))
  plt.imshow(ma.MaskedArray(label_image, mask), cmap=custom_cmap)
  plt.colorbar(orientation='horizontal')
  plt.title('Clustering result')
  plt.show()