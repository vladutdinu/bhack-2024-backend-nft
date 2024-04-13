import os
from fastapi import APIRouter, HTTPException, UploadFile, File
import numpy as np
import requests
from models.nft_model import NFTSimilarityResponse, NFTNsfwResponse, NFTUrl
from utils.chroma_db import get_query_chromadb
from utils.confidentiality_score import compute_confidence_scores
from utils.embedding import embed_image, get_image_embeddings, process_image
import pickle

nft_check_route = APIRouter()


@nft_check_route.post("/check_nft_similarity_bytes")
async def nft_check(nft: UploadFile = File(...)):
    content = await nft.read()
    query = get_query_chromadb(content)
    similar_nfts = query["documents"][0]
    scores = compute_confidence_scores(query["distances"][0])
    return NFTSimilarityResponse(
        filename=nft.filename, similar_nfts=similar_nfts, confidence_scores=scores
    )


@nft_check_route.post("/check_nft_nsfw_bytes")
async def nft_check_nsfw(nft: UploadFile = File(...)):
    content = await nft.read()

    model_names = ["knn", "svm", "rf"]
    model_paths = ["./trained_models_nsfw/" + f"{model}.pkl" for model in model_names]

    # Load each model
    try:
        models = [pickle.load(open(model_path, "rb")) for model_path in model_paths]
        knn, svm, rf = models  # Unpack the models into individual variables
        image_tensor = process_image(content)
        embeddings_list = get_image_embeddings(image_tensor)[0][0]

        # Use preloaded models for predictions
        votes = [
            models[idx].predict([embeddings_list])[0] for idx in range(len(model_names))
        ]

        # Majority vote
        final_prediction = 1 if np.sum(votes) > len(models) / 2 else 0

        return NFTNsfwResponse(filename=nft.filename, nsfw=final_prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@nft_check_route.post("/check_nft_similarity_url")
async def nft_check(nft: NFTUrl):

    response = requests.get(nft.url)

    query = get_query_chromadb(response.content)
    similar_nfts = query["documents"][0]
    scores = compute_confidence_scores(query["distances"][0])
    return NFTSimilarityResponse(
        filename=nft.url, similar_nfts=similar_nfts, confidence_scores=scores
    )


@nft_check_route.post("/check_nft_nsfw_url")
async def nft_check_nsfw(nft: NFTUrl):

    response = requests.get(nft.url)

    model_names = ["knn", "svm", "rf"]
    model_paths = ["./trained_models_nsfw/" + f"{model}.pkl" for model in model_names]

    # Load each model
    try:
        models = [pickle.load(open(model_path, "rb")) for model_path in model_paths]

        image_tensor = process_image(response.content)
        embeddings_list = get_image_embeddings(image_tensor)[0][0]

        # Use preloaded models for predictions
        votes = [
            models[idx].predict([embeddings_list])[0] for idx in range(len(model_names))
        ]

        # Majority vote
        final_prediction = 1 if np.sum(votes) > len(models) / 2 else 0

        return NFTNsfwResponse(filename=nft.url, nsfw=final_prediction)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
