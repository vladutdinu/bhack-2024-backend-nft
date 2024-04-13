"""Module providing """

from typing import List
from pydantic import BaseModel

class NFTSimilarityResponse(BaseModel):
    filename: str
    similar_nfts: List[str]
    confidence_scores: List[str]

class NFTNsfwResponse(BaseModel):
    filename: str
    nsfw: int

class NFTUrl(BaseModel):
    url:str