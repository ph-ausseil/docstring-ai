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
from docstring_ai.lib.config import MODEL , RETRY_BACKOFF


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
    match = code_pattern.search(message[-1].text.value)
    if match:
        return match.group(1)
    else:
        print(message[-1].text.value)
        raise Exception("No code block found in the assistant's response.")


def get_file_description(assistant_id : str, thread_id: str, file_content: str) -> str:
    """
    Sends the entire file content to the Assistant and retrieves a detailed description of the file.
    
    Args:
        assistant: The OpenAI Assistant object.
        thread_id (str): The ID of the thread to use for interaction.
        file_content (str): The complete content of the Python file.
        
    Returns:
        str: A detailed description of the file.
    """
    prompt = (
        "Provide a comprehensive & detailed description of the following Python file. "
        "Highlight its main functionalities, purpose, classes, and function constructors. "
        "Include any important details that would help understand the purpose, functionality, context, structure and intent of the code.\n\n"
        f"{file_content}"
    )
    
    try:
        message = openai.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=prompt,
        )

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
            return "Description unavailable due to missing response."


        # Assuming the last message is the assistant's response
        assistant_message = messages[-1].content
        if not messages:
            logging.error(f"No messages found in Thread: {thread_id}")
            return "Description unavailable due to missing response."

        return assistant_message[-1].text.value
    except Exception as e:

        # Retrieve the assistant's response
        thread = openai.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=current_run.id
            )
        thread_messages = openai.beta.threads.messages.list(thread_id=thread_id)
        messages = thread_messages.data
        if not messages:
            logging.error(f"No messages found in Thread: {thread_id}")
            return "Description unavailable due to missing response."

        # Assuming the last message is the assistant's response
        assistant_message = messages[-1].content
        if not assistant_message:
            logging.error("Assistant's message is empty.")
            return "Description unavailable due to empty response."

        return assistant_message[-1].text.value
    except Exception as e:
        logging.error(f"Error during docstring addition: {e}")
        return "Description unavailable due to an API error."


def add_docstrings_to_code( assistant_id: str, thread_id: str, code: str, context: str) -> str:
    """
    Sends code along with few-shot examples to the OpenAI Assistant to add docstrings.

    This function interacts with the OpenAI API to send the Python code and contextual examples,
    waits for the assistant's response, and returns the code with added docstrings.

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
        return final_code
    except Exception as e:
        logging.error("Attempting new run...")
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
