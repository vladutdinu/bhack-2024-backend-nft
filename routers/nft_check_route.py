from fastapi import APIRouter, UploadFile


nft_check_route = APIRouter()


@nft_check_route.post("/upload-nft")
async def nft_check(nft : UploadFile):
    return {"filename": nft.filename}