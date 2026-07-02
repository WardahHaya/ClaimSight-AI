import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from backend.config import CHROMA_DIR

COLLECTION_NAME = "claimsight_policy_rules"

_embeddings_instance = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Load the local sentence-transformers embedding model once per process.
    """
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
    return _embeddings_instance


def get_vectorstore(create_if_missing: bool = True) -> Chroma:
    """
    Return the persistent Chroma vector store instance.
    """
    if not create_if_missing and not os.path.exists(CHROMA_DIR):
        raise ValueError(
            f"Chroma directory {CHROMA_DIR} does not exist yet. Run `python backend/ingest.py --fetch` first."
        )

    os.makedirs(CHROMA_DIR, exist_ok=True)
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=get_embeddings(),
    )
