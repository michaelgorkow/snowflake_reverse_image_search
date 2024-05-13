import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import requests
from PIL import Image
from io import BytesIO
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def make_white_transparent(img):
    # Convert image to RGBA (if not already in that format)
    img = img.convert("RGBA")
    datas = img.getdata()
    
    newData = []
    for item in datas:
        # Change all white (also shades of whites)
        # pixels to transparent
        if item[0] >= 250 and item[1] >= 250 and item[2] >= 250:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)

    img.putdata(newData)
    return img

def download_image(url):
    response = requests.get(url)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        img = make_white_transparent(img)
        return img
    return None

def download_images(df):
    urls = df['PRESIGNED_URL'].tolist()
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(download_image, url) for url in urls]
        return [future.result() for future in futures]

def apply_pca(embeddings):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings)
    return pca_result

def normalize_pca_results(pca_result):
    scaler = MinMaxScaler()
    normalized_pca_result = scaler.fit_transform(pca_result)
    return normalized_pca_result

def plot_images_pca(pca_result, images):
    fig, ax = plt.subplots(figsize=(12, 12))
    for i, img in enumerate(images):
        if img:
            im = OffsetImage(img.resize((100, 100)), zoom=1)
            ab = AnnotationBbox(im, (pca_result[i, 0], pca_result[i, 1]), frameon=False)
            ax.add_artist(ab)
    ax.update_datalim(pca_result)
    ax.autoscale()
    plt.show()

def visualize_image_clusters(df):
    # Download images
    images = download_images(df)
    
    # Load embeddings and apply PCA
    embeddings = np.array(df['IMAGE_EMBEDDING'].tolist())
    pca_result = apply_pca(embeddings)

    # Normalize PCA results
    normalized_pca_result = normalize_pca_results(pca_result)
    
    # Plot PCA results with images
    plot_images_pca(normalized_pca_result, images)