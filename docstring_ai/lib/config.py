
# Constants
MODEL = "gpt-4o-mini"  # Replace with the appropriate model if necessary
MAX_TOKENS = 64000  # Maximum tokens per request
EMBEDDING_MODEL = "text-embedding-3-large"  # OpenAI embedding model
MAX_RETRIES = 5
RETRY_BACKOFF = 2  # seconds
CHROMA_COLLECTION_NAME = "python_file_contexts"
CACHE_FILE_NAME = "docstring_cache.json"  # Name of the cache file