from langchain_huggingface import HuggingFaceEmbeddings
from keybert import KeyBERT

_embedder = None
_keybert  = None

def get_embedder():
    global _embedder
    if _embedder:
        return _embedder
    _embedder = HuggingFaceEmbeddings (model_name="all-MiniLM-L6-v2")
    return _embedder

def get_keybert():
    global _keybert
    if _keybert:
        return _keybert
    _keybert = KeyBERT()
    return _keybert