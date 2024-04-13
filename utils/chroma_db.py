import chromadb
from utils.embedding import get_image_embeddings, process_image

def get_query_chromadb(content):
    client = chromadb.PersistentClient(path="./chromadb")
    collection = client.get_or_create_collection("nfts")
    image_tensor = process_image(content)
    embeddings_list = get_image_embeddings(image_tensor)
    return collection.query([embeddings_list[0][0]], n_results=3)