from fastapi import APIRouter, UploadFile, File
from models.nft_model import NFTResponse
from utils.chroma_db import get_query_chromadb
from utils.confidentiality_score import compute_confidence_scores
from utils.embedding import embed_image

nft_check_route = APIRouter()


@nft_check_route.post("/check-nft")
async def nft_check(nft: UploadFile = File(...)):
    content = await nft.read()
    query = get_query_chromadb(content)
    similar_nfts = query["documents"][0]
    scores = compute_confidence_scores(query["distances"][0])
    return NFTResponse(filename=nft.filename, similar_nfts=similar_nfts, confidence_scores=scores)
