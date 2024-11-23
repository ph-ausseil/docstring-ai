
import openai
import time
import ast
from typing import List, Dict
import logging
from docstring_ai import RETRY_BACKOFF
from docstring_ai.lib.prompt_utils import extract_code_from_message


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
        logging.error(f"Error parsing code for description: {e}")
    return "; ".join(descriptions)



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
        logging.error(f"Error extracting docstring for class '{class_name}': {e}")
    return ""


def add_docstrings_to_code(api_key: str, assistant_id: str, thread_id: str, code: str, context: str) -> str:
    """
    Send code along with few-shot examples to the Assistant to add docstrings.
    Returns the modified code.
    """
    try:
        # Add a message to the thread
        message = openai.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=(
                f"{context}\n\n"
                "Please add appropriate docstrings to the following Python code. "
                "Ensure that all functions, classes, and modules have clear and concise docstrings explaining their purpose, parameters, return values, and any exceptions raised.\n\n"
                "```python\n"
                f"{code}\n"
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
        return modified_code
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
