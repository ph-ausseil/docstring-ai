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
    logging.warning("Deprecated function : extract_description_from_docstrings , replaced by get_file_description")
    descriptions = []
    try:
        tree = ast.parse(code_with_docstrings)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                doc = ast.get_docstring(node)
                if doc:
                    name = "module" if isinstance(node, ast.Module) else node.name
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

def parse_classes(file_path: str) -> Dict[str, List[str]]:
    """
    Parse a Python file and return a dictionary of classes and their parent classes.

    This function reads a Python file and uses the Abstract Syntax Tree (AST) to
    identify classes and their inherited parent classes.

    Args:
        file_path (str): The path to the Python file to be parsed.

    Returns:
        Dict[str, List[str]]: A dictionary where keys are class names and values are lists of parent classes.

    Raises:
        Exception: If there is an error during file reading or parsing.
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
