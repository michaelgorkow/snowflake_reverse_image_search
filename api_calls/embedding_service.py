from PIL import Image
import numpy as np
import requests
from snowflake.snowpark.session import Session
import io

def get_snowflake_header_token(session):
    # need to change result format for token request
    session.sql(f"alter session set python_connector_query_result_format = json;").collect()
    token_data = session.connection._rest._token_request('ISSUE')
    api_session_token = token_data['data']['sessionToken']
    api_headers = {'Authorization': f'''Snowflake Token="{api_session_token}"'''}
    return api_headers


def get_embedding(session: Session, filename: str):
    # open image and read into memory
    img = Image.open(filename)
    inmemory_image_bytes = io.BytesIO()
    img.save(inmemory_image_bytes, 'jpeg')
    inmemory_image_bytes = inmemory_image_bytes.getvalue()
    # retrieve ingress-url
    IMAGE_CLASSIFICATION_SERVICE_URL = session.sql('SHOW ENDPOINTS IN SERVICE EMBEDDING_SERVICE').collect()[0]['ingress_url']
    IMAGE_CLASSIFICATION_SERVICE_URL = f'https://{IMAGE_CLASSIFICATION_SERVICE_URL}/generate-embeddings-api'
    # Call the embedding service
    response = requests.post(
        url=IMAGE_CLASSIFICATION_SERVICE_URL, 
        data=inmemory_image_bytes, 
        headers=get_snowflake_header_token(session)
        )
    embedding = np.array(response.json()['EMBEDDING'])
    # change result format back to arrow
    session.sql(f"alter session set python_connector_query_result_format = ARROW;").collect()
    return embedding