from snowflake.snowpark.session import Session
from snowflake.snowpark import functions as F
from snowflake.snowpark import types as T
import streamlit as st
from PIL import Image
import io
import matplotlib.pyplot as plt
import numpy as np
import requests
from snowflake.snowpark.session import Session
import os

st.set_page_config(layout="wide")

def get_embedding(img: Image):
    # open image and read into memory
    inmemory_image_bytes = io.BytesIO()
    img.save(inmemory_image_bytes, 'jpeg')
    inmemory_image_bytes = inmemory_image_bytes.getvalue()
    IMAGE_CLASSIFICATION_SERVICE_URL = "http://embedding-service:9000/generate-embeddings-api"
    response = requests.post(
        url=IMAGE_CLASSIFICATION_SERVICE_URL, 
        data=inmemory_image_bytes, 
        )
    embedding = np.array(response.json()['EMBEDDING'])
    return embedding

def plot_image_grid(dataframe, rel_path_col='RELATIVE_PATH', url_col='PRESIGNED_URL', cos_sim_col='COSINE_SIMILARITY', num_cols=5):
    # Define the number of columns for subplot
    num_rows = (len(dataframe) + num_cols - 1) // num_cols

    # Create a figure with subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 3 * num_rows))
    axes = axes.flatten()
    
    # Loop through the DataFrame rows
    for idx, row in dataframe.iterrows():
        # Get image from URL
        response = requests.get(row[url_col])
        img = Image.open(io.BytesIO(response.content))
        
        # Plot image
        axes[idx].imshow(img)
        axes[idx].set_title(f"{row[rel_path_col]}\nCosine Similarity: {row[cos_sim_col]:.2f}", fontsize=10)
        axes[idx].axis('off')
        
    # Hide unused subplots
    for idx in range(len(dataframe), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    return fig

def get_login_token():
  with open('/snowflake/session/token', 'r') as f:
    return f.read()
  
SNOWFLAKE_ACCOUNT = os.getenv('SNOWFLAKE_ACCOUNT')
SNOWFLAKE_HOST = os.getenv('SNOWFLAKE_HOST')
import os
snowflake_connection_cfg = {
    "HOST": os.getenv('SNOWFLAKE_HOST'),
    "ACCOUNT": os.getenv('SNOWFLAKE_ACCOUNT'),
    "TOKEN": get_login_token(),
    "AUTHENTICATOR": 'oauth',
    "ROLE": 'SPCS_ROLE',
    "DATABASE": 'REVERSE_IMAGE_SEARCH',
    "SCHEMA": 'PUBLIC',
    "WAREHOUSE": 'COMPUTE_WH'
}

# Creating Snowpark Session
session = Session.builder.configs(snowflake_connection_cfg).create()

st.title("Image Similarity App")

uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
st.sidebar.header('Query Image:')
if uploaded_file is not None:
    # Convert the file to an image
    image = Image.open(uploaded_file)
        
    # Display the image
    st.sidebar.image(image, caption='Uploaded Image.', use_column_width=True)

num_similar_images = st.slider('Number of similar images to retrieve',3,30)

st.header('Similar Images:')
# Retrieve the embedding
if uploaded_file is not None:
    embedding = get_embedding(img=image)
    # Search image database
    images_df = session.table('FASHION_IMAGES_EMBEDDINGS')
    images_df = images_df.with_column('PRESIGNED_URL', F.call_builtin('GET_PRESIGNED_URL', '@IMAGES', F.col('RELATIVE_PATH')))
    search_df = images_df.with_column('QUERY_EMBEDDING', F.lit(embedding[0].tolist()).cast(T.VectorType(float,768)))
    search_df = search_df.with_column('COSINE_SIMILARITY', F.call_builtin('VECTOR_COSINE_SIMILARITY', F.col('QUERY_EMBEDDING'), F.col('IMAGE_EMBEDDING')))
    search_df = search_df.order_by(F.col('COSINE_SIMILARITY').desc())
    search_df = search_df.with_column('COSINE_SIMILARITY', F.round('COSINE_SIMILARITY', 2))
    search_df[['RELATIVE_PATH','COSINE_SIMILARITY']].show(5)
    # Visualize the similar images
    fig = plot_image_grid(search_df.limit(num_similar_images).to_pandas())
    st.pyplot(fig)