# Constants

MODEL = "gpt-4o-mini"  
"""
str: The name of the model to be used for processing.
Replace with the appropriate model if necessary. This specifies which version of the model will be employed 
for inference and processing tasks.
"""

MAX_TOKENS = 64000  
"""
int: The maximum number of tokens allowed in a single request to the model.
This constant defines the upper limit on the length of the input text, ensuring that it does not exceed 
the capabilities of the model being used.
"""

EMBEDDING_MODEL = "text-embedding-3-large"  
"""
str: The name of the OpenAI embedding model used for converting text into embedding vectors.
This model is employed to generate numerical representations of the input text data, which can be 
used for various natural language processing tasks.
"""

MAX_RETRIES = 5  
"""
int: The maximum number of retry attempts for API requests.
This constant defines how many times the system should attempt to resend a request to an external service 
in case the first attempt fails due to a transient error.
"""

RETRY_BACKOFF = 3  
"""
int: The time, in seconds, to wait before retrying after a failed API request.
This constant is used to introduce a delay before a retry attempt, helping to mitigate the potential 
of overwhelming the server with rapid retry requests.
"""

CHROMA_COLLECTION_NAME = "python_file_contexts"  
"""
str: The name of the ChromaDB collection used to store context data.
This constant specifies the collection where context-related data is stored when interacting with 
the Chroma database, facilitating organized data retrieval.
"""

CACHE_FILE_NAME = "docstring_cache.json"  
"""
str: The name of the file used for caching purposes.
This constant defines the filename for storing cached data, allowing the application to save and 
retrieve data efficiently, enhancing performance.
"""
