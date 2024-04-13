import os
from utils.embedding import embed_image
import chromadb
import uuid
import time

file_path =  './nfts'

nft_files = [file for file in os.listdir(file_path) if file.endswith('.txt')]
nft_urls = []
for file in nft_files:
    with open(os.path.join(file_path, file), 'r') as nfts_txt_file:
        nfts = nfts_txt_file.read().split('\n')
        nft_urls.extend(nfts)

client = chromadb.PersistentClient(path="./chromadb")
collection = client.get_or_create_collection("nfts")
for url in nft_urls:
  time.sleep(1)
  collection.add([uuid.uuid4().hex], [embed_image(url)[0][0]], documents=[url], metadatas={"source":url})