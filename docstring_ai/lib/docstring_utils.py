"""
This module provides functions to extract descriptions from docstrings,
add docstrings to Python code using OpenAI’s Assistant, and parse
Python classes from a file. It includes utilities for interacting with
the OpenAI API and logging errors encountered during execution.

Functions:
- extract_description_from_docstrings: Extracts simple descriptions from docstrings in the given code.
- extract_class_docstring: Extracts the docstring of a specified class from the code.
- add_docstrings_to_code: Sends code to the OpenAI Assistant to add appropriate docstrings.
- parse_classes: Parses a Python file to extract a dictionary of classes and their parent classes.
"""

import openai
import time
import ast
from typing import List, Dict
import logging
from docstring_ai import RETRY_BACKOFF
from docstring_ai.lib.prompt_utils import extract_code_from_message

def extract_description_from_docstrings(code_with_docstrings: str) -> str:
    """
    Extracts a simple description from docstrings in the provided code.

    This function parses the code and retrieves the first line of docstrings
    for each function, class, and module, returning them in a formatted string.

    Args:
        code_with_docstrings (str): The Python code containing docstrings.

    Returns:
        str: A semicolon-separated string of descriptions extracted from 
             the docstrings. If no descriptions are found, returns an empty string.

    Raises:
        Exception: If there is an error while parsing the code.
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
        logging.error(f"Error parsing code for description: {e}")
    return "; ".join(descriptions)

def extract_class_docstring(code: str, class_name: str) -> str:
    """
    Extracts the docstring of a specific class from the provided code.

    Args:
        code (str): The Python code containing the class definition.
        class_name (str): The name of the class whose docstring is to be extracted.

    Returns:
        str: The docstring of the specified class, or an empty string if not found.

    Raises:
        Exception: If there is an error during class docstring extraction.
    """
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                doc = ast.get_docstring(node)
                return doc
    except Exception as e:
        logging.error(f"Error extracting docstring for class '{class_name}': {e}")
    return ""

def add_docstrings_to_code(api_key: str, assistant_id: str, thread_id: str, code: str, context: str) -> str:
    """
    Sends code along with few-shot examples to the OpenAI Assistant to add docstrings.

    Args:
        api_key (str): The API key for OpenAI.
        assistant_id (str): The ID of the assistant to use for the operation.
        thread_id (str): The ID of the thread for the conversation.
        code (str): The Python code for which docstrings need to be added.
        context (str): Contextual information or examples for the assistant.

    Returns:
        str: The modified code with added docstrings, or None if an error occurs.

    Raises:
        Exception: If there is an error during the interaction with the OpenAI API.
    """
    try:
        # Escape triple backticks in the original code to prevent interference
        escaped_code = code.replace('```', '` ``')
        # Add a message to the thread
        message = openai.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=(
                f"{context}\n\n"
                "Please add appropriate docstrings to the following Python code. "
                "Ensure that all functions, classes, and modules have clear and concise docstrings explaining their purpose, parameters, return values, and any exceptions raised.\n\n"
                "```python\n"
                f"{escaped_code}\n"
                "```"
            ),
        )
        logging.info(f"Message created with ID: {message.id} for Thread: {thread_id}")

        # Create a Run for the message
        run = openai.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
        )
        logging.info(f"Run created with ID: {run.id} for Thread: {thread_id}")

        # Poll for Run completion
        while True:
            current_run = openai.beta.threads.runs.retrieve(
                run_id=run.id,
                thread_id=thread_id
                )
            status = current_run.status
            if status == 'completed':
                logging.info(f"Run {run.id} completed.")
                break
            elif status in ['failed', 'expired', 'cancelled']:
                logging.error(f"Run {run.id} ended with status: {status}")
                return None
            else:
                logging.info(f"Run {run.id} status: {status}. Waiting for completion...")
                time.sleep(RETRY_BACKOFF)

        # Retrieve the assistant's response
        thread = openai.beta.threads.retrieve(thread_id=thread_id)
        thread_messages = openai.beta.threads.messages.list(
            thread_id=thread_id,
            order="asc")
        messages = thread_messages.data
        if not messages:
            logging.error(f"No messages found in Thread: {thread_id}")
            return None

        # Assuming the last message is the assistant's response
        assistant_message = messages[-1].content
        if not assistant_message:
            logging.error("Assistant's message is empty.")
            return None

        # Extract code block from assistant's message
        modified_code = extract_code_from_message(assistant_message)
        # Revert the escaped backticks to original
        final_code = modified_code.replace('` ``', '```')
        return modified_code
    except Exception as e:
        logging.error(f"Error during docstring addition: {e}")
        return None

        # Retrieve the assistant's response
        thread = openai.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=current_run.id
            )
        thread_messages = openai.beta.threads.messages.list(thread_id=thread_id)
        messages = thread_messages.data
        if not messages:
            logging.error(f"No messages found in Thread: {thread_id}")
            return None

        # Assuming the last message is the assistant's response
        assistant_message = messages[-1].content
        if not assistant_message:
            logging.error("Assistant's message is empty.")
            return None

        # Extract code block from assistant's message
        modified_code = extract_code_from_message(assistant_message)
        # Revert the escaped backticks to original
        final_code = modified_code.replace('` ``', '```')
        return final_code
    except Exception as e:
        logging.error(f"Error during docstring addition: {e}")
        return None


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
        logging.error(f"Error parsing classes in {file_path}: {e}")
    return classes
