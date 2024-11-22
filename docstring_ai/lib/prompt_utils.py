import openai
import chromadb
from docstring_ai.lib.chroma_utils import get_relevant_context
import logging
from typing import List, Dict
from docstring_ai.lib.config import MODEL


def initialize_assistant(api_key: str, assistant_name: str = "DocstringAssistant") -> str:
    """
    Initialize or retrieve an existing Assistant.
    Returns the Assistant ID.
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
            "Ensure that all functions, classes, and modules have clear and concise docstrings explaining their purpose, parameters, return values, and any exceptions raised."
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


def update_assistant_tool_resources(api_key: str, assistant_id: str, file_ids: List[str]):
    """
    Update the Assistant's tool_resources with the uploaded file IDs.
    """
    try:
        openai.beta.assistants.update(
            assistant_id=assistant_id,
            file_ids=file_ids
        )
        logging.info(f"Assistant '{assistant_id}' tool_resources updated with {len(file_ids)} files.")
    except Exception as e:
        logging.error(f"Error updating Assistant's tool_resources: {e}")


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
        logging.info(f"Thread created with ID: {thread['id']}")
        return thread['id']
    except Exception as e:
        logging.error(f"Error creating Thread: {e}")
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

