import os
import openai
import argparse
import time
import json
import ast
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import tiktoken
from typing import List, Dict
import hashlib

# Constants
MODEL = "gpt-4o-mini"  # Replace with the appropriate model if necessary
MAX_TOKENS = 64000  # Maximum tokens per request
EMBEDDING_MODEL = "text-embedding-3-large"  # OpenAI embedding model
MAX_RETRIES = 5
RETRY_BACKOFF = 2  # seconds
CHROMA_COLLECTION_NAME = "python_file_contexts"
CACHE_FILE_NAME = "docstring_cache.json"  # Name of the cache file

def get_python_files(repo_path: str) -> List[str]:
    """Recursively retrieve all Python files in the repository."""
    python_files = []
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files

def sort_files_by_size(file_paths: List[str]) -> List[str]:
    """Sort files in ascending order based on their file size."""
    sorted_files = sorted(file_paths, key=lambda x: os.path.getsize(x))
    print(f"Files sorted by size (ascending).")
    return sorted_files

def initialize_chroma() -> chromadb.Client:
    """Initialize ChromaDB client."""
    client = chromadb.Client(Settings(
        chroma_api_impl="rest",
        chroma_server_host="localhost",
        chroma_server_http_port="8000"
    ))
    return client

def get_or_create_collection(client: chromadb.Client, collection_name: str) -> chromadb.Collection:
    """Retrieve an existing collection or create a new one."""
    existing_collections = client.list_collections()
    for collection in existing_collections:
        if collection.name == collection_name:
            print(f"ChromaDB Collection '{collection_name}' found.")
            return client.get_collection(name=collection_name, embedding_function=embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai.api_key,
                model=EMBEDDING_MODEL
            ))
    print(f"ChromaDB Collection '{collection_name}' not found. Creating a new one.")
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai.api_key,
            model=EMBEDDING_MODEL
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
            print(f"Prepared file for embedding: {file_path}")
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    
    # Add to ChromaDB
    try:
        collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
        print(f"Embedded and stored {len(ids)} files in ChromaDB.")
    except Exception as e:
        print(f"Error adding documents to ChromaDB: {e}")

def parse_classes(file_path: str) -> Dict[str, List[str]]:
    """
    Parse a Python file and return a dictionary of classes and their parent classes.
    Example: {'ClassA': ['BaseClass1', 'BaseClass2'], ...}
    """
    classes = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=file_path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                parent_classes = [base.id if isinstance(base, ast.Name) else
                                  base.attr if isinstance(base, ast.Attribute) else
                                  "Unknown" for base in node.bases]
                classes[node.name] = parent_classes
    except Exception as e:
        print(f"Error parsing classes in {file_path}: {e}")
    return classes

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
                print("Reached maximum token limit for context.")
                return context
            context += doc + "\n\n"
            token_count += doc_tokens
    return context

def initialize_assistant(api_key: str, assistant_name: str = "DocstringAssistant") -> str:
    """
    Initialize or retrieve an existing Assistant.
    Returns the Assistant ID.
    """
    try:
        # List existing assistants
        response = openai.Assistant.list()
        assistants = response['data']
        for assistant in assistants:
            if assistant.get("name") == assistant_name:
                print(f"Assistant '{assistant_name}' found with ID: {assistant['id']}")
                return assistant['id']
        
        # If Assistant does not exist, create one
        instructions = (
            "You are an AI assistant specialized in adding comprehensive docstrings to Python code. "
            "Ensure that all functions, classes, and modules have clear and concise docstrings explaining their purpose, parameters, return values, and any exceptions raised."
        )
        assistant = openai.Assistant.create(
            name=assistant_name,
            description="Assistant to add docstrings to Python files.",
            model=MODEL,
            tools=[{"type": "code_interpreter"}, {"type": "file_search"}],
            tool_resources={
                "code_interpreter": {
                    "file_ids": []  # To be populated after uploading files
                },
                "file_search": {
                    "file_ids": []  # Similarly, if needed
                }
            },
            instructions=instructions
        )
        print(f"Assistant '{assistant_name}' created with ID: {assistant['id']}")
        return assistant['id']
    except Exception as e:
        print(f"Error initializing Assistant: {e}")
        return None

def update_assistant_tool_resources(api_key: str, assistant_id: str, file_ids: List[str]):
    """
    Update the Assistant's tool_resources with the uploaded file IDs.
    """
    try:
        # Fetch current assistant details
        assistant = openai.Assistant.retrieve(assistant_id)
        tool_resources = assistant.get("tool_resources", {})
        
        # Update 'code_interpreter' tool
        if "code_interpreter" not in tool_resources:
            tool_resources["code_interpreter"] = {}
        if "file_ids" not in tool_resources["code_interpreter"]:
            tool_resources["code_interpreter"]["file_ids"] = []
        tool_resources["code_interpreter"]["file_ids"].extend(file_ids)
        
        # Similarly, update 'file_search' tool if used
        if "file_search" in tool_resources:
            if "file_ids" not in tool_resources["file_search"]:
                tool_resources["file_search"]["file_ids"] = []
            tool_resources["file_search"]["file_ids"].extend(file_ids)
        
        # Update the Assistant with new tool_resources
        updated_assistant = openai.Assistant.modify(
            assistant_id,
            tool_resources=tool_resources
        )
        print(f"Assistant '{assistant_id}' tool_resources updated with {len(file_ids)} files.")
    except Exception as e:
        print(f"Error updating Assistant's tool_resources: {e}")

def create_thread(api_key: str, assistant_id: str, initial_messages: List[dict] = None) -> str:
    """
    Create a new Thread for the Assistant.
    Returns the Thread ID.
    """
    try:
        payload = {
            "assistant_id": assistant_id,
            "messages": initial_messages if initial_messages else []
        }
        thread = openai.Thread.create(**payload)
        print(f"Thread created with ID: {thread['id']}")
        return thread['id']
    except Exception as e:
        print(f"Error creating Thread: {e}")
        return None

def construct_few_shot_prompt(collection: chromadb.Collection, classes: Dict[str, List[str]], max_tokens: int) -> str:
    """
    Constructs a few-shot prompt using context summaries from ChromaDB.
    """
    context = get_relevant_context(collection, classes, max_tokens // 2)  # Allocate half tokens to context
    few_shot_examples = generate_few_shot_examples(context)
    prompt = few_shot_examples
    return prompt

def generate_few_shot_examples(context: str) -> str:
    """
    Generates few-shot examples based on the retrieved context summaries.
    """
    # For simplicity, assume context contains example docstrings
    # In a real scenario, you might format this differently
    examples = (
        "Here are some examples of Python classes with comprehensive docstrings:\n\n"
        f"{context}"
        "Now, please add appropriate docstrings to the following Python code:\n\n"
    )
    return examples

def add_docstrings_to_code(api_key: str, assistant_id: str, thread_id: str, code: str, context: str) -> str:
    """
    Send code along with few-shot examples to the Assistant to add docstrings.
    Returns the modified code.
    """
    try:
        # Create a Run with context
        run = openai.Run.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"{context}\n\n"
                        "Please add appropriate docstrings to the following Python code. "
                        "Ensure that all functions, classes, and modules have clear and concise docstrings explaining their purpose, parameters, return values, and any exceptions raised.\n\n"
                        "```python\n"
                        f"{code}\n"
                        "```"
                    )
                }
            ],
            max_prompt_tokens=5000,  # Adjust as needed
            max_completion_tokens=5000  # Adjust as needed
        )
        print(f"Run created with ID: {run['id']} for Thread: {thread_id}")
    
        # Poll for Run completion
        while True:
            current_run = openai.Run.retrieve(run['id'])
            status = current_run['status']
            if status == 'completed':
                print(f"Run {run['id']} completed.")
                break
            elif status in ['failed', 'expired', 'cancelled']:
                print(f"Run {run['id']} ended with status: {status}")
                return None
            else:
                print(f"Run {run['id']} status: {status}. Waiting for completion...")
                time.sleep(RETRY_BACKOFF)
    
        # Retrieve the assistant's response
        thread = openai.Thread.retrieve(thread_id)
        messages = thread.get('messages', [])
        if not messages:
            print(f"No messages found in Thread: {thread_id}")
            return None
    
        # Assuming the last message is the assistant's response
        assistant_message = messages[-1].get('content', "")
        if not assistant_message:
            print("Assistant's message is empty.")
            return None
    
        # Extract code block from assistant's message
        modified_code = extract_code_from_message(assistant_message)
        return modified_code
    except Exception as e:
        print(f"Error during docstring addition: {e}")
        return None

def extract_code_from_message(message: str) -> str:
    """
    Extracts the code block from the assistant's message.
    """
    import re
    code_pattern = re.compile(r"```python\n([\s\S]*?)```")
    match = code_pattern.search(message)
    if match:
        return match.group(1)
    else:
        raise Exception("No code block found in the assistant's response.")

def extract_description_from_docstrings(code_with_docstrings: str) -> str:
    """
    Extracts a simple description from docstrings.
    This can be enhanced with more sophisticated parsing.
    """
    descriptions = []
    try:
        tree = ast.parse(code_with_docstrings)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                doc = ast.get_docstring(node)
                if doc:
                    if isinstance(node, ast.Module):
                        name = "module"
                    else:
                        name = node.name
                    first_line = doc.strip().split('\n')[0]
                    descriptions.append(f"{name}: {first_line}")
    except Exception as e:
        print(f"Error parsing code for description: {e}")
    return "; ".join(descriptions)

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
        print(f"Stored summary for class '{class_name}' in ChromaDB.")
    except Exception as e:
        print(f"Error storing class summary for '{class_name}': {e}")

def compute_sha256(file_path: str) -> str:
    """
    Computes the SHA-256 hash of a file.
    """
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            # Read and update hash string value in blocks of 4K
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        print(f"Error computing SHA-256 for {file_path}: {e}")
        return ""

def load_cache(cache_file: str) -> Dict[str, str]:
    """
    Loads the cache from a JSON file.
    Returns a dictionary mapping file paths to their SHA-256 hashes.
    """
    if not os.path.exists(cache_file):
        print(f"No cache file found at '{cache_file}'. Starting with an empty cache.")
        return {}
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        print(f"Loaded cache with {len(cache)} entries.")
        return cache
    except Exception as e:
        print(f"Error loading cache file '{cache_file}': {e}")
        return {}

def save_cache(cache_file: str, cache: Dict[str, str]):
    """
    Saves the cache to a JSON file.
    """
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2)
        print(f"Cache saved with {len(cache)} entries.")
    except Exception as e:
        print(f"Error saving cache file '{cache_file}': {e}")

def main(repo_path: str, api_key: str):
    openai.api_key = api_key

    # Initialize ChromaDB
    print("\nInitializing ChromaDB...")
    chroma_client = initialize_chroma()
    collection = get_or_create_collection(chroma_client, CHROMA_COLLECTION_NAME)

    # Load cache
    cache_path = os.path.join(repo_path, CACHE_FILE_NAME)
    cache = load_cache(cache_path)

    # Step 1: Retrieve all Python files
    python_files = get_python_files(repo_path)
    print(f"Found {len(python_files)} Python files to process.")

    if not python_files:
        print("No Python files found. Exiting.")
        return

    # Step 2: Sort files by size (ascending)
    python_files_sorted = sort_files_by_size(python_files)

    # Step 3: Compute SHA-256 hashes and filter out unchanged files
    files_to_process = []
    for file_path in python_files_sorted:
        current_hash = compute_sha256(file_path)
        cached_hash = cache.get(os.path.relpath(file_path, repo_path))
        if current_hash == cached_hash:
            print(f"Skipping unchanged file: {file_path}")
        else:
            files_to_process.append(file_path)
    
    print(f"\n{len(files_to_process)} files to process after cache check.")

    if not files_to_process:
        print("No files need processing. Exiting.")
        return

    # Step 4: Embed and store files in ChromaDB
    print("\nEmbedding and storing Python files in ChromaDB...")
    embed_and_store_files(collection, files_to_process)

    # Step 5: Initialize Assistant
    print("\nInitializing Assistant...")
    assistant_id = initialize_assistant(api_key)
    if not assistant_id:
        print("Assistant initialization failed. Exiting.")
        return

    # Step 6: Update Assistant's tool_resources with uploaded file IDs
    print("\nUpdating Assistant's tool resources...")
    # Retrieve all file IDs from ChromaDB
    file_ids = [doc['id'] for doc in collection.get()['ids']]
    update_assistant_tool_resources(api_key, assistant_id, file_ids)

    # Step 7: Create a Thread
    print("\nCreating a new Thread...")
    thread_id = create_thread(api_key, assistant_id)
    if not thread_id:
        print("Thread creation failed. Exiting.")
        return

    # Step 8: Process Each Python File
    print("\nProcessing Python files to add docstrings...")
    context_summary = []
    for idx, file_path in enumerate(files_to_process, 1):
        print(f"\nProcessing file {idx}/{len(files_to_process)}: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_code = f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            continue

        # Parse classes and identify dependencies
        classes = parse_classes(file_path)
        if not classes:
            print(f"No classes found in {file_path}. Skipping context retrieval.")
            context = ""
        else:
            # Retrieve relevant context summaries from ChromaDB
            context = get_relevant_context(collection, classes, max_tokens=MAX_TOKENS // 2)  # Allocate half tokens to context
            print(f"Retrieved context with {tiktoken.get_encoding('gpt2').encode(context).__len__()} tokens.")

        # Construct few-shot prompt
        few_shot_prompt = construct_few_shot_prompt(collection, classes, max_tokens=MAX_TOKENS)

        # Add docstrings using Assistants API
        modified_code = add_docstrings_to_code(api_key, assistant_id, thread_id, original_code, few_shot_prompt)

        if modified_code and modified_code != original_code:
            try:
                # Backup original file
                backup_path = f"{file_path}.bak"
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_code)
                print(f"Backup of original file created at {backup_path}")

                # Update the file with modified code
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(modified_code)
                print(f"Updated docstrings in {file_path}")

                # Extract description from docstrings
                description = extract_description_from_docstrings(modified_code)
                context_summary.append({
                    "file": os.path.relpath(file_path, repo_path),
                    "description": description
                })

                # Parse classes again to extract summaries
                modified_classes = parse_classes(file_path)
                for class_name in modified_classes.keys():
                    # Extract the docstring for each class
                    class_docstring = extract_class_docstring(modified_code, class_name)
                    if class_docstring:
                        summary = class_docstring.strip().split('\n')[0]  # First line as summary
                        store_class_summary(collection, os.path.relpath(file_path, repo_path), class_name, summary)
                
                # Update cache with new hash
                new_hash = compute_sha256(file_path)
                cache[os.path.relpath(file_path, repo_path)] = new_hash
                print(f"Updated cache for file: {file_path}")

            except Exception as e:
                print(f"Error updating file {file_path}: {e}")
        else:
            print(f"No changes made to {file_path}.")
            # Update cache even if no changes to prevent reprocessing unchanged files
            current_hash = compute_sha256(file_path)
            cache[os.path.relpath(file_path, repo_path)] = current_hash

    # Step 9: Save Context Summary
    context_summary_path = os.path.join(repo_path, "context_summary.json")
    try:
        with open(context_summary_path, "w", encoding='utf-8') as f:
            json.dump(context_summary, f, indent=2)
        print(f"\nDocstring generation completed. Context summary saved to '{context_summary_path}'.")
    except Exception as e:
        print(f"Error saving context summary: {e}")

    # Step 10: Save Cache
    save_cache(cache_path, cache)

def extract_class_docstring(code: str, class_name: str) -> str:
    """
    Extracts the docstring of a specific class from the modified code.
    """
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                doc = ast.get_docstring(node)
                return doc
    except Exception as e:
        print(f"Error extracting docstring for class '{class_name}': {e}")
    return ""

def compute_sha256(file_path: str) -> str:
    """
    Computes the SHA-256 hash of a file.
    """
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            # Read and update hash string value in blocks of 4K
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        print(f"Error computing SHA-256 for {file_path}: {e}")
        return ""

def load_cache(cache_file: str) -> Dict[str, str]:
    """
    Loads the cache from a JSON file.
    Returns a dictionary mapping file paths to their SHA-256 hashes.
    """
    if not os.path.exists(cache_file):
        print(f"No cache file found at '{cache_file}'. Starting with an empty cache.")
        return {}
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        print(f"Loaded cache with {len(cache)} entries.")
        return cache
    except Exception as e:
        print(f"Error loading cache file '{cache_file}': {e}")
        return {}

def save_cache(cache_file: str, cache: Dict[str, str]):
    """
    Saves the cache to a JSON file.
    """
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2)
        print(f"Cache saved with {len(cache)} entries.")
    except Exception as e:
        print(f"Error saving cache file '{cache_file}': {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add docstrings to Python files in a repository using OpenAI Assistants API and ChromaDB with caching.")
    parser.add_argument("repo_path", help="Path to the repository directory.")
    parser.add_argument("--api_key", help="OpenAI API key. Can also be set via OPENAI_API_KEY environment variable.")

    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key must be provided via --api_key or OPENAI_API_KEY environment variable.")
        exit(1)

    main(args.repo_path, api_key)
