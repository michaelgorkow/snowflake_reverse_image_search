# Snowflake Reverse Image Search

This repository showcases how to build a Reverse Image Search in Snowflake.  
You'll find code for:  

1. **Setting Up Snowflake Objects**: Creating databases, schemas, warehouses, compute pools and services using Snowflake's Python API.
2. **Uploading a Hugging Face Dataset**: Retrieving images from a Hugging Face dataset and upload them to a Snowflake stage.
3. **Generating Image Embeddings**: Using GPU accelerated Snowpark Container Services with a Hugging Face Image Embedding Model.
4. **Visualizing Embedding Results**: Use Principal Component Analysis to visualize image clusters.
5. **Streamlit App**: Simple but intuitive Streamlit App to upload an image and return similar images.

This repository is part of a blog article which can be found here:  
[Reverse Image Search in Snowflake](https://medium.com/@michaelgorkow/e666d173adb0?source=friends_link&sk=5fd577cc00636b94a03c14fccb72dcdc)

## Requirements

To run this pipeline, you will need to install:

- [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html)
- [Docker](https://www.docker.com/products/docker-desktop/)

You will also need access to a Snowflake Account with Snowpark Container Services enabled:
- [Free Snowflake Trial Account](https://signup.snowflake.com/)

Your Snowflake role should have relevant privileges.  
I used the following code for this demo to create a new role:

```sql
USE ROLE ACCOUNTADMIN;
CREATE ROLE SPCS_ROLE;

GRANT ALL ON ACCOUNT TO ROLE SPCS_ROLE;
GRANT ROLE SPCS_ROLE TO USER ADMIN;
```
Note: Providing all privileges is not recommended for production.

## Setup

1. Clone the repository:

   ```sh
   git clone https://github.com/michaelgorkow/snowflake_reverse_image_search.git
   ```

2. Navigate to the project directory:

   ```sh
   cd snowflake_reverse_image_search
   ```

3. Create a fresh Conda Environment:
   ```sh
   conda env create -f conda_env.yml
   ```

4. Activate Conda Environment:
   ```sh
   conda activate pysnowpark_reverse_image_search
   ```

5. Run the ```reverse_image_search.ipynb``` notebook until you have set up all the Snowflake objects. (especially the image registry)


5. Build and upload the containers:
   ```sh
   docker build --platform linux/amd64 -t org-account.registry.snowflakecomputing.com/reverse_image_search/public/image_repository/dinov2_base:latest .
   docker push org-account.registry.snowflakecomputing.com/reverse_image_search/public/image_repository/dinov2_base:latest

   docker build --platform linux/amd64 -t org-account.registry.registry.snowflakecomputing.com/reverse_image_search/public/image_repository/image_similarity_app:latest .
   docker push org-account.registry.snowflakecomputing.com/reverse_image_search/public/image_repository/image_similarity_app:latest
   ```

5. Continue with the ```reverse_image_search.ipynb``` notebook.

6. Open the Streamlit App using the link generated in the notebook.