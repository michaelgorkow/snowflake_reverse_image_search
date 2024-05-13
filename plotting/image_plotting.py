import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

# Define a function to load an image from a URL
def load_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

def plot_similar_images(df, file_left='RELATIVE_PATH_LEFT', url_left='PRESIGNED_URL_LEFT', 
                        file_right='RELATIVE_PATH_RIGHT', url_right='PRESIGNED_URL_RIGHT', 
                        similarity='COSINE_SIMILARITY', RETURN_TOP_N=5):
    # Setup the plot
    fig, axs = plt.subplots(nrows=len(df[url_left].unique()), ncols=RETURN_TOP_N+1, figsize=(20, 10))

    for i, (index, group) in enumerate(df.groupby(url_left)):
        # Load the image from the left URL
        img_left = load_image(group.iloc[0][url_left])
        axs[i, 0].imshow(img_left)
        axs[i, 0].set_title(f"{group.iloc[0][file_left]}\nQuery Image")
        axs[i, 0].axis('off')

        # Load up to RETURN_TOP_N images from the right URL
        for j, row in enumerate(group.nlargest(RETURN_TOP_N, similarity).itertuples()):
            img_right = load_image(getattr(row, url_right))
            axs[i, j + 1].imshow(img_right)
            axs[i, j + 1].set_title(f"{getattr(row, file_right)}\nSimilarity:{getattr(row, similarity):.2f}")
            axs[i, j + 1].axis('off')

    plt.tight_layout()
    plt.show()

def plot_image_grid(dataframe, rel_path_col='RELATIVE_PATH', url_col='PRESIGNED_URL', cos_sim_col='COSINE_SIMILARITY'):
    # Define the number of columns for subplot
    num_cols = 5
    num_rows = (len(dataframe) + num_cols - 1) // num_cols

    # Create a figure with subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 3 * num_rows))
    axes = axes.flatten()

    # Loop through the DataFrame rows
    for idx, row in dataframe.iterrows():
        # Get image from URL
        response = requests.get(row[url_col])
        img = Image.open(BytesIO(response.content))
        
        # Plot image
        axes[idx].imshow(img)
        axes[idx].set_title(f"{row[rel_path_col]}\nCosine Similarity: {row[cos_sim_col]:.2f}", fontsize=10)
        axes[idx].axis('off')
        
        # Hide empty plots if any
        if idx >= len(dataframe):
            axes[idx].axis('off')

    plt.tight_layout()
    plt.show()