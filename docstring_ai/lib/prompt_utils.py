"""
This module provides functionalities to initialize and manage an AI assistant
for adding docstrings to Python code. It utilizes OpenAI's API to create and
interact with the assistant, manage threads, and construct prompts based on
context from ChromaDB.

Functions:
- initialize_assistant: Initialize or retrieve an existing assistant.
- update_assistant_tool_resources: Update the assistant's resources with file IDs.
- create_thread: Create a new thread for the assistant's interaction.
- construct_few_shot_prompt: Constructs a few-shot prompt using context summaries.
- generate_few_shot_examples: Generates few-shot examples based on context.
- extract_code_from_message: Extracts code blocks from the assistant's messages.
"""

import openai
import chromadb
from docstring_ai.lib.chroma_utils import get_relevant_context
import logging
from typing import List, Dict
from docstring_ai.lib.config import MODEL


def initialize_assistant(api_key: str, assistant_name: str = "DocstringAssistant") -> str:
    """
    Initialize or retrieve an existing Assistant.

    This function checks for existing assistants by name and returns the
    assistant ID if found. If not, it creates a new assistant with specified
    instructions.

    Args:
        api_key (str): The API key for OpenAI authentication.
        assistant_name (str): The name of the assistant to retrieve or create. Default is "DocstringAssistant".

    Returns:
        str: The ID of the Assistant, or None if an error occurred.

    Raises:
        Exception: If there is an error in retrieving or creating the Assistant.
    """
    try:
        # List existing assistants
        response = openai.beta.assistants.list()
        assistants = response.data
        for assistant in assistants:
            if assistant.name == assistant_name:
                logging.info(f"Assistant '{assistant_name}' found with ID: {assistant.id}")
                return assistant.id

        # If Assistant does not exist, create one
        instructions = (
            "You are an AI assistant specialized in adding comprehensive docstrings to Python code. "
            "Ensure that all functions, classes, and modules have clear and docstrings. "
            "Docstrings should give extensive context and explain purpose, parameters, return values, and any exceptions raised."
        )
        assistant = openai.beta.assistants.create(
            name=assistant_name,
            description="Assistant to add docstrings to Python files.",
            model=MODEL,
            tools=[{"type": "code_interpreter"}, {"type": "file_search"}],
            instructions=instructions
        )
        logging.info(f"Assistant '{assistant_name}' created with ID: {assistant.id}")
        return assistant.id
    except Exception as e:
        logging.error(f"Error initializing Assistant: {e}")
        return None


def update_assistant_tool_resources(api_key: str, assistant_id: str, file_ids: List[str]) -> None:
    """
    Update the Assistant's tool_resources with the uploaded file IDs.

    This function creates a vector store for the files and updates the assistant's
    tool resources to include the new vector store.

    Args:
        api_key (str): The API key for OpenAI authentication.
        assistant_id (str): The ID of the assistant to update.
        file_ids (List[str]): A list of file IDs to add to the assistant's resources.

    Raises:
        Exception: If there is an error while updating the assistant's resources.
    """
    try:
        vector_store = openai.beta.vector_stores.create(name=f"Docstring-AI::{assistant_id}")
        vector_store_file_batch = openai.beta.vector_stores.file_batches.create( 
            vector_store_id=vector_store.id,
            file_ids=file_ids)
        openai.beta.assistants.update(
            assistant_id=assistant_id,
            tool_resources={
                "file_search": {
                    "vector_store_ids": [vector_store.id]
                }
            }
        )
        logging.info(f"Assistant '{assistant_id}' tool_resources updated with {len(file_ids)} files.")
    except Exception as e:
        logging.error(f"Error updating Assistant's tool_resources: {e}")


def create_thread(api_key: str, assistant_id: str, initial_messages: List[dict] = None) -> str:
    """
    Create a new Thread for the Assistant.

    A thread is used for maintaining a conversation context with the assistant.

    Args:
        api_key (str): The API key for OpenAI authentication.
        assistant_id (str): The ID of the assistant for which to create a thread.
        initial_messages (List[dict], optional): A list of initial messages to start the thread.

    Returns:
        str: The ID of the created thread, or None if an error occurred.

    Raises:
        Exception: If there is an error creating the thread.
    """
    try:
        payload = {
            "messages": initial_messages if initial_messages else []
        }
        thread = openai.beta.threads.create(**payload)
        logging.info(f"Thread created with ID: {thread.id}")
        return thread.id
    except Exception as e:
        logging.error(f"Error creating Thread: {e}")
        return None


def construct_few_shot_prompt(collection: chromadb.Collection, classes: Dict[str, List[str]], max_tokens: int) -> str:
    """
    Constructs a few-shot prompt using context summaries from ChromaDB.

    This function retrieves relevant context based on class dependencies and
    constructs a prompt for the assistant.

    Args:
        collection (chromadb.Collection): The ChromaDB collection to query for context.
        classes (Dict[str, List[str]]): A dictionary containing class names and their parent classes.
        max_tokens (int): The maximum number of tokens to be used in the prompt.

    Returns:
        str: The constructed few-shot prompt.

    Raises:
        Exception: If there is an error retrieving context or generating the prompt.
    """
    context = get_relevant_context(collection, classes, max_tokens // 2)  # Allocate half tokens to context
    prompt = ""
    if context: 
        few_shot_examples = generate_few_shot_examples(context)
        prompt = few_shot_examples
    return prompt


def generate_few_shot_examples(context: str) -> str:
    """
    Generates few-shot examples based on the retrieved context summaries.

    The context is assumed to contain example docstrings, and this function
    formats them into a prompt suitable for the assistant.

    Args:
        context (str): The context containing example docstrings.

    Returns:
        str: The few-shot examples formatted in a readable manner.
    """
    # For simplicity, assume context contains example docstrings
    # In a real scenario, you might format this differently
    examples = (
        "Here are some examples of Python classes with comprehensive docstrings:\n\n"
        f"{context}"
        "Now, please add appropriate docstrings to the following Python code:\n\n"
    )
    return examples


def extract_code_from_message(message: str) -> str:
    """
    Extracts the code block from the assistant's message.
    """
    import re
    code_pattern = re.compile(r"```python\n([\s\S]*?)```")
    match = code_pattern.search(message[-1].text.value)
    if match:
        return match.group(1)
    else:
        raise Exception("No code block found in the assistant's response.")

