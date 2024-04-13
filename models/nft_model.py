"""Module providing """

from typing import List
from pydantic import BaseModel

class NFTResponse(BaseModel):
    filename: str
    similar_nfts: List[str]
    confidence_scores: List[str]
