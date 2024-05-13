import logging
import sys
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import concurrent.futures
from io import BytesIO
import requests
import os
import io
import json
from PIL import Image
os.environ['HF_HOME'] = '/models'
model_id = os.getenv('MODEL')
from transformers import AutoImageProcessor, AutoModel

# Logging
def get_logger(logger_name):
   logger = logging.getLogger(logger_name)
   logger.setLevel(logging.DEBUG)
   handler = logging.StreamHandler(sys.stdout)
   handler.setLevel(logging.DEBUG)
   handler.setFormatter(
      logging.Formatter(
      '%(name)s [%(asctime)s] [%(levelname)s] %(message)s'))
   logger.addHandler(handler)
   return logger
logger = get_logger('snowpark-container-service')

app = FastAPI()

logger.info(f'cuda.is_available(): {torch.cuda.is_available()}')
logger.info(f'cuda.device_count(): {torch.cuda.device_count()}')
logger.info('Loading Model ...')
processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)
logger.info('Finished Loading Model.')


def generate_embedding(images):
    """Generate embeddings from a list of images."""
    with torch.no_grad():
        inputs = processor(images=images, return_tensors="pt")
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        embedding_vectors = last_hidden_states.mean(dim=1).numpy().tolist()
        return embedding_vectors

def download_image(url):
    """Download an image from a URL."""
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return image

def download_images(urls):
    """Download multiple images in parallel from a list of URLs."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        images = list(executor.map(download_image, urls))
    return images
   
@app.post("/generate-embeddings", tags=["Endpoints"])
async def generate_embeddings(request: Request):
    request_body = await request.json()
    request_body = request_body['data']
    return_data = []
    row_ids = [item[0] for item in request_body]
    logger.info(f'BATCHSIZE:{len(row_ids)}')
    image_urls = [item[1] for item in request_body]
    images = download_images(image_urls)
    embeddings = generate_embedding(images)
    return_data = [[element1, element2] for element1, element2 in zip(row_ids, embeddings)]
    return {"data": return_data}
    
    
@app.post("/generate-embeddings-api", tags=["Endpoints"])
async def generate_embeddings_api2(request: Request):
    image_bytes = await request.body()
    img = Image.open(io.BytesIO(image_bytes))
    embedding = generate_embedding(img)
    return {'EMBEDDING':embedding}