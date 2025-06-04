import streamlit as st
import os
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern
from skimage.transform import resize
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Configuración inicial
st.set_page_config(layout="wide")
st.title("Buscador de Películas por Similitud Visual")

# Cargar dataset y PCA
df = pd.read_csv('features_all_rgb_lbp.csv')
filenames = df['filename'].values
X = df.select_dtypes(include=[np.number]).values

# Entrenar PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Función para extraer características de imagen
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
    lbp_hist, _ = np.histogram(
        lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2), density=True
    )
    return np.concatenate([hist_rgb, lbp_hist])

# Subida o selección de imagen
st.sidebar.header("Selecciona un póster")
opcion = st.sidebar.radio("¿Cómo quieres ingresar una imagen?", ["Subir una imagen", "Elegir una imagen del dataset"])

if opcion == "Subir una imagen":
    uploaded_file = st.sidebar.file_uploader("Cargar imagen", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        input_img = imread(uploaded_file)
        st.image(input_img, caption="Imagen cargada", use_column_width=True)
elif opcion == "Elegir una imagen del dataset":
    imagen_sel = st.sidebar.selectbox("Selecciona un archivo", filenames)
    input_img = imread(os.path.join('posters', imagen_sel))
    st.image(input_img, caption=f"Imagen: {imagen_sel}", use_column_width=True)

# Buscar similares
if 'input_img' in locals():
    input_feat = extract_features(input_img).reshape(1, -1)
    input_pca = pca.transform(input_feat)
    distancias = np.linalg.norm(X_pca - input_pca, axis=1)
    indices_similares = np.argsort(distancias)[:5]

    st.subheader("Resultados: Películas más similares")
    cols = st.columns(5)
    for i, idx in enumerate(indices_similares):
        with cols[i]:
            st.image(os.path.join('posters', filenames[idx]), use_column_width=True, caption=f"Simil #{i+1}")
