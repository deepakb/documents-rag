from fastapi import APIRouter, Depends

from services.github_handler import GithubHandler
from vendor.github import get_github_handler

router = APIRouter()


@router.post("/github/", tags=["github"], summary="Process github repo to embed")
async def add_github_documents(
    repo_url: str,
    github_handler: GithubHandler = Depends(get_github_handler)
):
    response = await github_handler.process(repo_url)
    return response
