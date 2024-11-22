import os
import openai
import argparse
import time
import json
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import tiktoken
from typing import List, Dict
import hashlib
from dotenv import load_dotenv
from datetime import datetime
from github import Github, GithubException
import subprocess
import sys
import logging
import difflib
from .config import EMBEDDING_MODEL

def initialize_chroma() -> chromadb.Client:
    """Initialize ChromaDB client."""
    client = chromadb.Client(Settings(
        chroma_server_host="localhost",
        chroma_server_http_port="8000"
    ))
    return client


def get_or_create_collection(client: chromadb.Client, collection_name: str) -> chromadb.Collection:
    """Retrieve an existing collection or create a new one."""
    existing_collections = client.list_collections()
    for collection in existing_collections:
        if collection.name == collection_name:
            logging.info(f"ChromaDB Collection '{collection_name}' found.")
            return client.get_collection(
                name=collection_name,
                embedding_function=embedding_functions.OpenAIEmbeddingFunction(
                    api_key=openai.api_key,
                    #model=EMBEDDING_MODEL
                )
            )
    logging.info(f"ChromaDB Collection '{collection_name}' not found. Creating a new one.")
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai.api_key,
            #model=EMBEDDING_MODEL
        )
    )
    return collection



def embed_and_store_files(collection: chromadb.Collection, python_files: List[str]):
    """Embed each Python file and store in ChromaDB."""
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
    Ensures the total tokens do not exceed max_tokens.
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



def store_class_summary(collection: chromadb.Collection, file_path: str, class_name: str, summary: str):
    """
    Stores the class summary in ChromaDB for future context.
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

