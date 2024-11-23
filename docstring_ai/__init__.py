"""
This module imports constants related to the model configuration for processing tasks.
These constants include information on model names, token limits, embedding models,
API request limits, caching, and database collections.

Constants:
    MODEL (str): The name of the model to be used for processing.
    MAX_TOKENS (int): The maximum number of tokens allowed in a single request to the model.
    EMBEDDING_MODEL (str): The model used for embedding text data.
    MAX_RETRIES (int): The maximum number of retry attempts for API requests.
    RETRY_BACKOFF (int): The time, in seconds, to wait before retrying after a failed API request.
    CHROMA_COLLECTION_NAME (str): The name of the ChromaDB collection used to store context data.
    CACHE_FILE_NAME (str): The name of the file used for caching purposes.
"""

from .lib.config import (
    MODEL,  # str: The name of the model to be used for processing.
    MAX_TOKENS,  # int: The maximum number of tokens allowed in a single request to the model.
    EMBEDDING_MODEL,  # str: The model used for embedding text data.
    MAX_RETRIES,  # int: The maximum number of retry attempts for API requests.
    RETRY_BACKOFF,  # int: The time, in seconds, to wait before retrying after a failed API request.
    CHROMA_COLLECTION_NAME,  # str: The name of the ChromaDB collection used to store context data.
    CACHE_FILE_NAME  # str: The name of the file used for caching purposes.
)
