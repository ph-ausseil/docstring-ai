"""
This module provides functions to extract descriptions from docstrings,
add docstrings to Python code using OpenAI’s Assistant, and parse
Python classes from a file. It includes utilities for interacting with
the OpenAI API and logging errors encountered during execution.

Functions:
- extract_description_from_docstrings: Extracts simple descriptions from docstrings in the given code.
- extract_class_docstring: Extracts the docstring of a specified class from the code.
- add_docstrings: Sends code to the OpenAI Assistant to add appropriate docstrings.
- parse_classes: Parses a Python file to extract a dictionary of classes and their parent classes.
"""

import openai
import time
import ast
import logging
from typing import List, Dict
from docstring_ai.lib.logger import show_file_progress
from docstring_ai import RETRY_BACKOFF
from docstring_ai.lib.prompt_utils import extract_code_from_message

def extract_description_from_docstrings(code_with_docstrings: str) -> str:
    """
    Extracts simple descriptions from docstrings in the provided code.

    This function parses the code and retrieves the first line of docstrings
    for each function, class, and module, returning them in a formatted string.

    Args:
        code_with_docstrings (str): The Python code containing docstrings.

    Returns:
        str: A semicolon-separated string of descriptions extracted from 
             the docstrings. If no descriptions are found, returns an empty string.

    Raises:
        Exception: If there is an error while parsing the code or extracting descriptions.
    """
    logging.warning("Deprecated function: extract_description_from_docstrings, replaced by generate_file_description")
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

    This function analyzes the code to find the class definition matching the 
    specified class name and returns its associated docstring.

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
    Parses a Python file and returns a dictionary of classes and their parent classes.

    This function reads a Python file and uses the Abstract Syntax Tree (AST) 
    to identify classes and their inherited parent classes. The result is a 
    dictionary where class names are keys and their corresponding parent classes 
    are values in a list.

    Args:
        file_path (str): The path to the Python file to be parsed.

    Returns:
        Dict[str, List[str]]: A dictionary where keys are class names and values are lists of parent classes.

    Raises:
        Exception: If there is an error during file reading or parsing.
    """
    logging.warning(f"Deprecated : Get imported elements from `list_imports_from_package`")
    logging.info(f"Parsing classes from : {file_path}")
    classes = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
        
        tree = ast.parse(file_content, filename=file_path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                parent_classes = [
                    base.id if isinstance(base, ast.Name) else
                    base.attr if isinstance(base, ast.Attribute) else
                    "Unknown" for base in node.bases
                ]
                classes[node.name] = parent_classes
    except Exception as e:

        print(file_content)
        print("#######################")
        print("#########EOF###########")
        print("#######################")
        logging.error(f"Error parsing classes in {file_path}: {e}")
    return classes

import ast
import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import ast
import logging
from typing import Dict, List, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocstringExtractor:
    """
    A class to extract all docstrings from a Python file and list imports from a specified package,
    and compile them into a readable format.
    """

    def __init__(self, file_path: str):
        """
        Initializes the DocstringExtractor with the path to the Python file.

        Args:
            file_path (str): The path to the Python script to be analyzed.
        """
        self.file_path = file_path
        self.file_content: Optional[str] = None
        self.tree: Optional[ast.AST] = None
        self.docstrings: Dict[str, Dict[str, str]] = {}
        self.imports: Dict[str, List[str]] = {}

    def read_file(self) -> None:
        """
        Reads the content of the Python file.

        Raises:
            FileNotFoundError: If the file does not exist.
            IOError: If there is an error reading the file.
        """
        logger.info(f"Reading file: {self.file_path}")
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.file_content = f.read()
            logger.debug("File read successfully.")
        except FileNotFoundError:
            logger.error(f"File not found: {self.file_path}")
            raise
        except IOError as e:
            logger.error(f"IO error when reading file {self.file_path}: {e}")
            raise

    def parse_ast(self) -> None:
        """
        Parses the file content into an Abstract Syntax Tree (AST).

        Raises:
            SyntaxError: If the file content contains invalid Python syntax.
            ValueError: If file_content is not set.
        """
        if self.file_content is None:
            logger.error("File content is not loaded. Call read_file() first.")
            raise ValueError("File content is not loaded.")

        logger.info("Parsing AST.")
        try:
            self.tree = ast.parse(self.file_content, filename=self.file_path)
            logger.debug("AST parsed successfully.")
        except SyntaxError as e:
            logger.error(f"Syntax error in file {self.file_path}: {e}")
            raise

    def extract_docstrings(self) -> None:
        """
        Extracts all docstrings from the AST and populates the docstrings dictionary.

        The dictionary structure:
        {
            'element_name': { 'type': 'function' | 'class' | 'async function' | 'module', 'docstring': '...' },
            ...
        }
        For methods within classes, the key is 'ClassName.method_name'.
        """
        if self.tree is None:
            logger.error("AST is not parsed. Call parse_ast() first.")
            raise ValueError("AST is not parsed.")

        logger.info("Extracting docstrings.")
        # Extract module-level docstring
        module_docstring = ast.get_docstring(self.tree)
        if module_docstring:
            self.docstrings['module'] = {'type': 'module', 'docstring': module_docstring}
            logger.debug("Module docstring extracted.")

        def _extract(element: ast.AST, parent_name: Optional[str] = None) -> None:
            """
            Recursively extracts docstrings from AST nodes.

            Args:
                element (ast.AST): The current AST node.
                parent_name (Optional[str]): The fully qualified name of the parent element.
            """
            for node in ast.iter_child_nodes(element):
                if isinstance(node, ast.ClassDef):
                    class_name = node.name
                    qualified_name = f"{parent_name}.{class_name}" if parent_name else class_name
                    class_doc = ast.get_docstring(node)
                    if class_doc:
                        self.docstrings[qualified_name] = {'type': 'class', 'docstring': class_doc}
                        logger.debug(f"Class docstring extracted for '{qualified_name}'.")
                    # Recursively extract from the class
                    _extract(node, qualified_name)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_type = 'async function' if isinstance(node, ast.AsyncFunctionDef) else 'function'
                    func_name = node.name
                    qualified_name = f"{parent_name}.{func_name}" if parent_name else func_name
                    func_doc = ast.get_docstring(node)
                    if func_doc:
                        self.docstrings[qualified_name] = {'type': func_type, 'docstring': func_doc}
                        logger.debug(f"{func_type.capitalize()} docstring extracted for '{qualified_name}'.")
                    # Recursively extract from the function (e.g., nested functions)
                    _extract(node, qualified_name)
                # Add other element types if needed (e.g., modules within packages)

        # Start extracting from the module level
        _extract(self.tree)

        logger.info(f"Total docstrings extracted: {len(self.docstrings)}")

    def list_imports_from_package(self, package: str) -> List[str]:
        """
        Extracts and lists all imported names from the specified package within the Python script.

        Args:
            package (str): The package name to extract imports from (e.g., 'docstring_ai.lib').

        Returns:
            List[str]: A list of names imported from the specified package.

        Raises:
            ValueError: If the AST is not parsed.
        """
        if self.tree is None:
            logger.error("AST is not parsed. Call parse_ast() first.")
            raise ValueError("AST is not parsed.")

        logger.info(f"Starting to parse imports from '{package}' in file: {self.file_path}")
        imported_names: List[str] = []

        for node in ast.walk(self.tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module
                if module is None:
                    # Handles cases like 'from . import something'
                    logger.debug(f"Skipping relative import in file {self.file_path}.")
                    continue

                # Check if the module matches the target package
                if module == package or module.startswith(f"{package}."):
                    for alias in node.names:
                        if alias.name == '*':
                            logger.warning(f"Wildcard import detected in {self.file_path} from {module}. Skipping.")
                            continue
                        imported_names.append(alias.name)
                        logger.debug(f"Imported '{alias.name}' from '{module}'.")

        logger.info(f"Total imports found from '{package}': {len(imported_names)}")
        return imported_names

    def compile(self) -> str:
        """
        Compiles the extracted docstrings into a readable text format.

        Returns:
            str: The compiled docstrings in a readable text format.
        """
        logger.info("Compiling docstrings into readable text.")
        compiled_text = ""
        for element, info in self.docstrings.items():
            compiled_text += f"{element} ({info['type']}):\n{info['docstring']}\n\n"
            logger.debug(f"Compiled docstring for '{element}'.")

        compiled_text = compiled_text.strip()  # Remove trailing whitespace
        logger.info("Compilation complete.")
        return compiled_text

    def get_docstrings_dict(self) -> Dict[str, Dict[str, str]]:
        """
        Returns the dictionary of extracted docstrings.

        Returns:
            Dict[str, Dict[str, str]]: The dictionary mapping element names to their types and docstrings.
        """
        return self.docstrings

    def process(self) -> Dict[str, Dict[str, str]]:
        """
        High-level method to perform the complete docstring extraction process.

        Returns:
            Dict[str, Dict[str, str]]: The dictionary of extracted docstrings.
        """
        try:
            self.read_file()
            self.parse_ast()
            self.extract_docstrings()
            return self.docstrings
        except Exception as e:
            logger.error(f"Failed to extract docstrings: {e}")
            return {}

    def process_imports(self, package: str) -> List[str]:
        """
        High-level method to perform the import listing process for a specified package.

        Args:
            package (str): The package name to extract imports from.

        Returns:
            List[str]: A list of names imported from the specified package.
        """
        try:
            # Ensure the file is read and AST is parsed
            if self.file_content is None:
                self.read_file()
            if self.tree is None:
                self.parse_ast()
            imports = self.list_imports_from_package(package)
            self.imports[package] = imports
            return imports
        except Exception as e:
            logger.error(f"Failed to list imports from package '{package}': {e}")
            return []
