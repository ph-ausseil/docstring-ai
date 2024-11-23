# Constants

MODEL = "gpt-4o-mini"  
"""
str: The name of the model to be used for processing tasks.

This constant specifies which version of the model will be employed for inference and processing tasks.
It can be replaced with the appropriate model name based on the specific use case or model availability.

Usage:
    Set this constant to the desired model version before processing tasks.
"""

MAX_TOKENS = 64000  
"""
int: The maximum number of tokens allowed in a single request to the model.

This constant defines the upper limit on the length of the input text, ensuring that it does not exceed 
the processing capabilities of the model being used. This helps prevent errors related to input size and 
optimizes the model's processing performance.

Usage:
    Ensure that input text length does not exceed this limit to avoid errors during processing.
"""

EMBEDDING_MODEL = "text-embedding-3-large"  
"""
str: The name of the OpenAI embedding model used for converting text into embedding vectors.

This constant specifies the model utilized to generate numerical representations of the input text data, 
which are necessary for various natural language processing tasks such as text classification, semantic 
similarity, or clustering.

Usage:
    Use this model when embedding text data for any subsequent NLP tasks requiring embeddings.
"""

MAX_RETRIES = 5  
"""
int: The maximum number of retry attempts for API requests.

This constant defines how many times the system should attempt to resend a request to an external service 
if the first attempt fails due to a transient error. This is beneficial for improving the resilience of 
the application against temporary network issues or service interruptions.

Usage:
    Modify this value to adjust the application's tolerance for transient errors during API interactions.
"""

RETRY_BACKOFF = 3  
"""
int: The time, in seconds, to wait before retrying after a failed API request.

This constant specifies the delay introduced before making a retry attempt following a failed request. 
Introducing this backoff time helps to avoid overwhelming the server with rapid retry requests, thus enabling 
better handling of rate limits and server load management.

Usage:
    Adjust this duration to control how quickly the application should attempt to retry failed requests.
"""

CHROMA_COLLECTION_NAME = "python_file_contexts"  
"""
str: The name of the ChromaDB collection used to store context data.

This constant indicates the collection within a Chroma database where context-related data is stored when 
interacting with the database system. It facilitates organized data retrieval and management of persistent 
context throughout the application lifecycle.

Usage:
    Use this name when accessing or manipulating the ChromaDB collection for context storage.
"""

CACHE_FILE_NAME = "docstring_cache.json"  
"""
str: The name of the file used for caching purposes.

This constant specifies the filename for storing cached data, which allows the application to efficiently 
save and retrieve data. Utilizing caching enhances performance by avoiding redundant computations or 
data retrievals, thereby reducing latency and improving the responsiveness of operations.

Usage:
    Utilize this filename when implementing caching logic to store and retrieve data efficiently.
"""

CONTEXT_SUMMARY_PATH = "context_summary.json"
"""
str: The path for storing context summaries.

This constant specifies the file path where context summaries are stored, enabling easy retrieval and 
management of context data for subsequent processing tasks. It ensures that important context information 
is preserved and can be accessed when needed.

Usage:
    Save context summaries to this path for future reference or processing tasks.
"""
