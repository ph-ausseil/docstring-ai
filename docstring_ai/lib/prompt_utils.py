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
import time
import openai
import chromadb
from docstring_ai.lib.chroma_utils import get_relevant_context
import logging
from typing import List, Dict
from docstring_ai.lib.config import MODEL, RETRY_BACKOFF


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
        response = openai.beta.assistants.list()
        for assistant in response.data:
            if assistant.name == assistant_name:
                logging.info(f"Assistant '{assistant_name}' found with ID: {assistant.id}")
                return assistant.id

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
    Update the Assistant's tool resources with the uploaded file IDs.

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
        vector_store_id = create_vector_store(f"Docstring-AI::{assistant_id}", file_ids)
        openai.beta.assistants.update(
            assistant_id=assistant_id,
            tool_resources={
                "file_search": {
                    "vector_store_ids": [vector_store_id]
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
    try:
        context = get_relevant_context(collection, classes, max_tokens // 2)
        if not context:
            return ""
        examples = (
            "Here are some examples of Python classes with comprehensive docstrings:\n\n"
            f"{context}"
            "Now, please add appropriate docstrings to the following Python code:\n\n"
        )
        return examples
    except Exception as e:
        logging.error(f"Error constructing few-shot prompt: {e}")
        return ""



def extract_code_from_message(message: str) -> str:
    """
    Extracts the code block from the assistant's message.

    This function uses a regular expression to find and extract code blocks
    formatted in a specific way from the assistant's message.

    Args:
        message (str): The assistant's message string containing the code block.

    Returns:
        str: The extracted code block.

    Raises:
        Exception: If no code block is found in the assistant's response.
    """
    import re
    code_pattern = re.compile(r"```python\n([\s\S]*?)```")
    match = code_pattern.search(message)
    if match:
        return match.group(1)
    else:
        raise Exception("No code block found in the assistant's response.")


def send_message_to_assistant(assistant_id: str, thread_id: str, prompt: str) -> str:
    """
    Sends a prompt to the Assistant and retrieves the response.

    Args:
        assistant_id (str): The ID of the Assistant.
        thread_id (str): The ID of the thread for communication.
        prompt (str): The prompt or content to send to the Assistant.

    Returns:
        str: The Assistant's response text, or an error message if an issue occurs.
    """
    try:
        message = openai.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=prompt,
        )
        run = openai.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
        )
        if poll_run_completion(
            run_id=  run.id, 
            thread_id=thread_id
            ):
            return retrieve_last_assistant_message(thread_id)[-1].text.value
        return "Operation failed due to incomplete run."
    except Exception as e:
        logging.error(f"Error during interaction with Assistant: {e}")
        return "Operation failed due to an API error."

def generate_file_description(assistant_id: str, thread_id: str, file_content: str) -> str:
    """
    Generates a detailed description of a Python file using the Assistant.

    Args:
        assistant_id (str): The ID of the Assistant.
        thread_id (str): The ID of the thread for communication.
        file_content (str): The content of the Python file.

    Returns:
        str: A detailed description of the file.
    """
    prompt = (
        "Provide a comprehensive & detailed description of the following Python file. "
        "Highlight its main functionalities, purpose, classes, and function constructors. "
        "Include any important details that would help understand the purpose, functionality, context, structure, and intent of the code.\n\n"
        "```python\n"
        f"{file_content}\n"
        "```"
    )
    return send_message_to_assistant(assistant_id, thread_id, prompt)

def add_docstrings(assistant_id: str, thread_id: str, code: str, context: str) -> str:
    """
    Adds docstrings to Python code using the Assistant.

    Args:
        assistant_id (str): The ID of the Assistant.
        thread_id (str): The ID of the thread for communication.
        code (str): The Python code to process.
        context (str): Contextual examples or instructions for generating docstrings.

    Returns:
        str: The code with added docstrings, or None if an error occurs.
    """
    escaped_code = code.replace('```', '` ``')
    prompt = (
        "Please add appropriate docstrings to the following Python code. "
        "Ensure that all functions, classes, and modules have clear and concise docstrings explaining their purpose, parameters, return values, and any exceptions raised.\n\n"
        "```python\n"
        f"{escaped_code}\n"
        "```"
    )
    if context:
        prompt = f"{context}\n\n" + prompt

    try :
        response = send_message_to_assistant(assistant_id, thread_id, prompt)
    except : 
        print(f"Issue parssing the message {response}")
        raise Exception(e)

    if response:
        return extract_code_from_message(response).replace('` ``', '```')
    return None


# Utility Functions

def create_vector_store(vector_store_name: str, file_ids: List[str]) -> str:
    """
    Creates a vector store and associates it with file IDs.

    Args:
        vector_store_name (str): Name for the vector store.
        file_ids (List[str]): List of file IDs to associate with the vector store.

    Returns:
        str: The ID of the created vector store.
    """
    vector_store = openai.beta.vector_stores.create(name=vector_store_name)
    openai.beta.vector_stores.file_batches.create(
        vector_store_id=vector_store.id,
        file_ids=file_ids
    )
    return vector_store.id


def poll_run_completion(run_id: str, thread_id: str) -> bool:
    """
    Polls until the run is completed, failed, or cancelled.

    Args:
        run_id (str): The ID of the run to monitor.
        thread_id (str): The thread ID associated with the run.

    Returns:
        bool: True if the run completed successfully, False otherwise.
    """
    while True:
        current_run = openai.beta.threads.runs.retrieve(
            run_id=run_id,
            thread_id=thread_id
        )
        status = current_run.status
        if status == 'completed':
            logging.info(f"Run {run_id} completed.")
            return True
        elif status in ['failed', 'expired', 'cancelled']:
            logging.error(f"Run {run_id} ended with status: {status}")
            return False
        else:
            logging.info(f"Run {run_id} status: {status}. Waiting for completion...")
            time.sleep(RETRY_BACKOFF)


def retrieve_last_assistant_message(thread_id: str) -> str:
    """
    Retrieves the last message from a thread.

    Args:
        thread_id (str): The thread ID from which to retrieve the message.

    Returns:
        str: The last message content, or None if no message is found.
    """
    thread_messages = openai.beta.threads.messages.list(
        thread_id=thread_id,
        order="asc"
    ).data
    if not thread_messages:
        logging.error(f"No messages found in Thread: {thread_id}")
        return None


    print(f"#######################\n")
    print(f"role:{thread_messages[-1].role}\n")
    print(f"create_at:{thread_messages[-1].created_at}\n")
    print(f"status:{thread_messages[-1].status}\n\n")

    try:
        extract_code_from_message(thread_messages[-1].content)
    except Exception:
        for message in thread_messages:
            print(f"#######################\n")
            print(f"role:{message.role}\n")
            print(f"create_at:{message.created_at}\n")
            print(f"status:{message.status}\n")
    return thread_messages[-1].content

