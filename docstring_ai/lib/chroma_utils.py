import os
import openai
from docstring_ai.lib.logger import show_file_progress
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import tiktoken
from typing import List, Dict
from docstring_ai import EMBEDDING_MODEL


def initialize_chroma() -> chromadb.Client:
    """Initialize ChromaDB client.

    This function establishes a connection to the ChromaDB server.

    Returns:
        chromadb.Client: A ChromaDB client instance connected to the server.

    Example:
        client = initialize_chroma()
    """
    client = chromadb.Client(Settings(
        chroma_server_host="localhost",
        chroma_server_http_port="8000"
    ))
    return client


def get_or_create_collection(client: chromadb.Client, collection_name: str) -> chromadb.Collection:
    """Retrieve an existing collection or create a new one.

    This function checks if a specified collection exists in ChromaDB and returns it.
    If the collection does not exist, it creates a new one with the given name.

    Args:
        client (chromadb.Client): The ChromaDB client used to interact with the database.
        collection_name (str): The name of the collection to retrieve or create.

    Returns:
        chromadb.Collection: The ChromaDB collection instance.

    Raises:
        Exception: If there is an issue retrieving or creating the collection.
    """
    existing_collections = client.list_collections()
    
    for collection in existing_collections:
        if collection.name == collection_name:
            logging.info(f"ChromaDB Collection '{collection_name}' found.")
            return client.get_collection(
                name=collection_name,
                embedding_function=embedding_functions.OpenAIEmbeddingFunction(
                    api_key=openai.api_key,
                    model_name=EMBEDDING_MODEL
                )
            )
    
    logging.info(f"ChromaDB Collection '{collection_name}' not found. Creating a new one.")
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai.api_key,
            model_name=EMBEDDING_MODEL
        )
    )
    return collection


def embed_and_store_files(collection: chromadb.Collection, python_files: List[str]) -> None:
    """Embed each Python file and store it in ChromaDB.

    This function reads the contents of each specified Python file, embeds the content,
    and stores the embedded representations in the ChromaDB collection.

    Args:
        collection (chromadb.Collection): The ChromaDB collection where documents will be stored.
        python_files (List[str]): A list of file paths to the Python files to be embedded.

    Raises:
        Exception: If there's an error reading the files or adding them to ChromaDB.
    """
    ids = []
    documents = []
    metadatas = []
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            doc_id = os.path.relpath(file_path)
            ids.append(doc_id)
            documents.append(content)
            metadatas.append({"file_path": file_path})
            logging.info(f"Prepared file for embedding: {file_path}")
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}")

    # Add to ChromaDB
    try:
        collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
        logging.info(f"Embedded and stored {len(ids)} files in ChromaDB.")
    except Exception as e:
        logging.error(f"Error adding documents to ChromaDB: {e}")


def get_relevant_context(collection: chromadb.Collection, classes: Dict[str, List[str]], max_tokens: int) -> str:
    """
    Retrieve relevant documents from ChromaDB based on class dependencies.

    This function fetches relevant document content from the specified collection
    while ensuring that the total token count does not exceed the specified maximum.

    Args:
        collection (chromadb.Collection): The ChromaDB collection to query.
        classes (Dict[str, List[str]]): A dictionary mapping class names to their dependencies.
        max_tokens (int): The maximum number of tokens allowed for the retrieved context.

    Returns:
        str: The accumulated context as a single string.

    Example:
        context = get_relevant_context(collection, classes, max_tokens)
    """
    encoder = tiktoken.get_encoding("gpt2")
    context = ""
    token_count = 0
    for class_name, parents in classes.items():
        query = f"{class_name} " + " ".join(parents)
        results = collection.query(
            query_texts=[query],
            n_results=5  # Adjust based on desired breadth
        )
        for doc in results['documents'][0]:
            doc_tokens = len(encoder.encode(doc))
            if token_count + doc_tokens > max_tokens:
                logging.info("Reached maximum token limit for context.")
                return context
            context += doc + "\n\n"
            token_count += doc_tokens
    return context


def store_class_summary(collection: chromadb.Collection, file_path: str, class_name: str, summary: str) -> None:
    """
    Store the class summary in ChromaDB for future context.

    This function embeds the provided summary for a specific class and stores it 
    in the specified ChromaDB collection, associating it with the respective 
    file path and class name.

    Args:
        collection (chromadb.Collection): The ChromaDB collection where the summary will be stored.
        file_path (str): The path to the file containing the class.
        class_name (str): The name of the class for which the summary is stored.
        summary (str): The summary text to be embedded and stored.

    Raises:
        Exception: If there's an error storing the class summary in ChromaDB.
    
    Example:
        store_class_summary(collection, file_path, class_name, summary)
    """
    try:
        doc_id = f"{file_path}::{class_name}"
        collection.add(
            documents=[summary],
            ids=[doc_id],
            metadatas=[{"file_path": file_path, "class_name": class_name}]
        )
        logging.info(f"Stored summary for class '{class_name}' in ChromaDB.")
    except Exception as e:
        logging.error(f"Error storing class summary for '{class_name}': {e}")
