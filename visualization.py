import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px


def tsne_visualization(embeddings_file):
    data = pd.read_csv(embeddings_file)

    # Calculate the t-SNE representation of the embeddings
    X = np.vstack(data['embedding'].apply(eval))

    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # Create a mapping of categories to colors
    unique_categories = data['category'].unique()
    category_color_map = {category: plt.cm.jet(i / len(unique_categories)) for i, category in
                          enumerate(unique_categories)}
    colors = [category_color_map[category] for category in data['category']]
    rgb_colors = [(r, g, b) for r, g, b, _ in colors]

    plt.figure(figsize=(10, 8))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=rgb_colors, alpha=0.5)

    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')

    file_name_parts = embeddings_file.split("_")
    title = 't-SNE Visualization of Document Embeddings by Category\n{} Embeddings'.format(
        file_name_parts[-1].split(".")[0].title())
    plt.title(title)

    for category, color in category_color_map.items():
        plt.scatter([], [], color=color, label=category)

    # plt.legend()
    # plt.legend(loc='upper left', bbox_to_anchor=(0.8, 1))
    plt.legend(loc="upper left", framealpha=0.0)
    plt.show()


def tsne_visualization_3d(embeddings_file):
    data = pd.read_csv(embeddings_file)

    # Calculate the t-SNE representation of the embeddings
    X = np.vstack(data['embedding'].apply(eval))

    tsne = TSNE(n_components=3, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # Create a mapping of categories to colors
    unique_categories = data['category'].unique()
    category_color_map = {category: plt.cm.jet(i / len(unique_categories)) for i, category in
                          enumerate(unique_categories)}
    colors = [category_color_map[category] for category in data['category']]
    rgb_colors = [(r, g, b) for r, g, b, _ in colors]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=rgb_colors, alpha=0.5)

    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_zlabel('t-SNE Dimension 3')

    file_name_parts = embeddings_file.split("_")
    title = 't-SNE 3D Visualization of Document Embeddings by Category\n{} Embeddings'.format(
        file_name_parts[-1].split(".")[0].title())
    plt.title(title)

    for category, color in category_color_map.items():
        ax.scatter([], [], [], color=color, label=category)

    ax.legend()
    plt.show()


def tsne_visualization_3d_interactive(embeddings_file):
    data = pd.read_csv(embeddings_file)

    # Calculate the t-SNE representation of the embeddings
    X = np.vstack(data['embedding'].apply(eval))

    tsne = TSNE(n_components=3, random_state=42)
    X_tsne = tsne.fit_transform(X)

    data['tsne_dimension_1'] = X_tsne[:, 0]
    data['tsne_dimension_2'] = X_tsne[:, 1]
    data['tsne_dimension_3'] = X_tsne[:, 2]

    fig = px.scatter_3d(data, x='tsne_dimension_1', y='tsne_dimension_2', z='tsne_dimension_3',
                        color='category', hover_data=['category'], opacity=0.7)

    file_name_parts = embeddings_file.split("_")
    title = 'Interactive 3D t-SNE Visualization of Document Embeddings by Category\n{} Embeddings'.format(
        file_name_parts[-1].split(".")[0].title())
    fig.update_layout(title=title)
    fig.show()


def pca_visualization(embeddings_file):
    data = pd.read_csv(embeddings_file)

    # Calculate the PCA representation of the embeddings
    X = np.vstack(data['embedding'].apply(eval))

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Create a mapping of categories to colors
    unique_categories = data['category'].unique()
    category_color_map = {category: plt.cm.jet(i / len(unique_categories)) for i, category in
                          enumerate(unique_categories)}
    colors = [category_color_map[category] for category in data['category']]
    rgb_colors = [(r, g, b) for r, g, b, _ in colors]

    plt.figure(figsize=(10, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=rgb_colors, alpha=0.5)

    plt.xlabel('PCA Dimension 1')
    plt.ylabel('PCA Dimension 2')

    file_name_parts = embeddings_file.split("_")
    title = 'PCA Visualization of Document Embeddings by Category\n{} Embeddings'.format(
        file_name_parts[-1].split(".")[0].title())
    plt.title(title)

    for category, color in category_color_map.items():
        plt.scatter([], [], color=color, label=category)

    plt.legend()
    plt.show()
