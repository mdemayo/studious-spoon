import streamlit as st
import os
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern
from skimage.transform import resize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# 🔁 Mostrar representantes por cluster
def mostrar_representantes(X_pca, labels_kmeans, centroids, filepaths, titles, posters_folder='posters', top_n=3):
    st.subheader("También podrían interesarte...")
    unique_clusters = np.unique(labels_kmeans)
    for cluster_id in unique_clusters:
        st.markdown(f"### Grupo de pelis {cluster_id}:")
        cluster_indices = np.where(labels_kmeans == cluster_id)[0]
        cluster_points = X_pca[cluster_indices]
        centroide = centroids[cluster_id]
        distancias = np.linalg.norm(cluster_points - centroide, axis=1)
        indices_ordenados = cluster_indices[np.argsort(distancias)[:top_n]]
        cols = st.columns(top_n)
        for i, idx in enumerate(indices_ordenados):
            with cols[i]:
                ruta = os.path.join(posters_folder, filepaths[idx])
                st.image(ruta, caption=titles[idx], use_container_width=True)

# Configuración inicial
st.set_page_config(layout="wide")
st.title("Bienvenid@ a tu app de Recomendación de Pelis")

# Cargar datos
df = pd.read_csv('features_all_rgb_lbp.csv')
posters_df = pd.read_csv('posters.csv', encoding='latin1')

# Unir con títulos
df['imdbId'] = df['filename'].str.replace('.jpg', '', regex=False).astype(int)
df = df.merge(posters_df[['imdbId', 'Title']], on='imdbId', how='left')
df['Title'] = df['Title'].fillna(df['filename'])
titles = df['Title'].values
filepaths = df['filename'].values
X = df.select_dtypes(include=[np.number]).drop(columns=['imdbId']).values

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Función de extracción de características
def extract_features(image):
    image_resized = resize(image, (128, 128))
    gray = rgb2gray(image_resized)
    hist_rgb = np.concatenate([
        np.histogram(image_resized[:, :, i], bins=16, range=(0, 1), density=True)[0]
        for i in range(3)
    ])
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2), density=True)
    return np.concatenate([hist_rgb, lbp_hist])

# Subida o selección
st.sidebar.header("Selecciona una película")
opcion = st.sidebar.radio("¿Cómo quieres elegir la peli?", ["Subir un póster", "Elegir una película"])

if opcion == "Subir un póster":
    uploaded_file = st.sidebar.file_uploader("Cargar imagen", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        input_img = imread(uploaded_file)
        st.image(input_img, caption="Imagen cargada", use_container_width=True)

elif opcion == "Elegir una película":
    imagen_sel = st.sidebar.selectbox("Elige una peli de la lista", titles)
    idx_sel = np.where(titles == imagen_sel)[0][0]
    input_img = imread(os.path.join('posters', filepaths[idx_sel]))
    st.image(input_img, caption=f"Película: {titles[idx_sel]}", use_container_width=True)

# Buscar similares
if 'input_img' in locals():
    input_feat = extract_features(input_img).reshape(1, -1)
    input_pca = pca.transform(input_feat)
    distancias = np.linalg.norm(X_pca - input_pca, axis=1)
    indices_similares = np.argsort(distancias)[:5]

    st.subheader("Resultados: Te recomendamos estas películas...")
    cols = st.columns(5)
    for i, idx in enumerate(indices_similares):
        with cols[i]:
            st.image(os.path.join('posters', filepaths[idx]), use_container_width=True, caption=titles[idx])

    # Mostrar representantes por cluster
    st.markdown("---")
    with st.expander("Otros títulos que podrían interesarte:"):
        kmeans = KMeans(n_clusters=3, random_state=42).fit(X_pca)
        labels_kmeans = kmeans.labels_
        centroids = kmeans.cluster_centers_
        mostrar_representantes(X_pca, labels_kmeans, centroids, filepaths, titles)

#para ejecutar: streamlit run app.py

