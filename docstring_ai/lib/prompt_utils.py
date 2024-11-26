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
from openai.types.beta import vector_store_create_params
import chromadb
from docstring_ai.lib.chroma_utils import get_relevant_context
import logging
from typing import List, Dict, Callable
from docstring_ai.lib.config import MODEL, RETRY_BACKOFF, setup_logging, MAX_RETRIES
from pydantic import BaseModel, Field
import json

setup_logging()

ASSISTANTS_DEFAULT_TOOLS = [
                {"type": "code_interpreter"}, 
                {"type": "file_search"},
            ]
class PythonFile(BaseModel):
    new_file_content: str = Field(description="Updated python script with the updated docstrings.")

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
                logging.debug(f"Assistant '{assistant_name}' found with ID: {assistant.id}")
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
            tools=ASSISTANTS_DEFAULT_TOOLS,
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
        logging.debug(f"Assistant '{assistant_id}' tool_resources updated with {len(file_ids)} files.")
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


def construct_few_shot_prompt(
    collection: chromadb.Collection,
    classes: Dict[str, List[str]],
    max_tokens: int,
    context: str = None
    ) -> str:
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
        documents = get_relevant_context(collection=collection,
        classes =classes,
        max_tokens = max_tokens // 2,
        where={"file_type": "script"},
        )

        examples = "You will be asked to generate dosctrings. To do so we will give you some example of python code as well as some contextual information.\n"

        if documents:    
            examples += "Python classes with comprehensive docstrings:\n\n"
            examples +=f"{documents}\n\n"
        if context: 
            examples +="Contextual informations\n"
            examples +=f"{context}\n\n"
        
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


def send_message_to_assistant(
    assistant_id: str,
    thread_id: str,
    prompt: str,
    response_format: BaseModel = None,
    tools : List = [],
    tool_choice = "auto",
    functions : Dict[str, Callable] = {}
    ) -> str:
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
            response_format=response_format,
            tool_choice=tool_choice ,
            tools=ASSISTANTS_DEFAULT_TOOLS + tools 
            )
        if poll_run_completion(
            run_id=  run.id, 
            thread_id=thread_id,
            functions=functions
            ):
            last_assistant_message = retrieve_last_assistant_message(thread_id)
            return last_assistant_message[-1].text.value
        return "Operation failed due to incomplete run."
    except IndexError as e:
        print(f"last_assistant_message : {last_assistant_message}")
        raise e
    except Exception as e:
        print(f"Response format is : {response_format}")
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
    return send_message_to_assistant(
        assistant_id = assistant_id, thread_id = thread_id, prompt = prompt)

def create_file_with_docstring(
    assistant_id: str,
    thread_id: str,
    code: str,
    context: str,
    functions : Dict[str, Callable]
    ) -> str:
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
    prompt = str(context) + (
        "Please add appropriate docstrings to the following Python code. "
        "Ensure that all functions, classes, and modules have clear and concise docstrings explaining their purpose, parameters, return values, and any exceptions raised.\n\n"
        "```python\n"
        f"{escaped_code}\n"
        "```"
    )
    if context:
        prompt = f"{context}\n\n" + prompt

    try :
        response = send_message_to_assistant(
            assistant_id = assistant_id,
            thread_id = thread_id, 
            prompt = prompt,
            tool_choice= {"type": "function", "function": {"name": "write_file_with_new_docstring"}},
            tools = [
                {"type": "function",
                    "function": {
                        "name": "write_file_with_new_docstring",
                        "description": "Writes the updated Python file content with updated docstrings.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "new_file_content": {
                                    "type": "string",
                                    "description": "The complete content of the Python file with the updated docstrings."
                                }
                            },
                            "required": ["new_file_content"]
                        }
                    }
                }
            ],
            functions= functions
            )
        # response_format = {
        #     'type': 'json_schema',
        #     'json_schema' : {
        #         'name': 'new_pyton_script',
        #         'schema': {
        #             'type': 'object',
        #             'properties': {
        #                 'content': {'type': 'string', 'description': 'Updated python script with the updated docstrings.'}
        #                 },
        #             'required': ['content'],
        #             'additionalProperties': False
        #             },
        #         'strict': True}
        #         }
        
    except Exception as e: 
        print(f"Error : {e}")
        raise Exception(e)

    if response:
        logging.info(f"The Response is : {response}")
        return response
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
    vector_store = openai.beta.vector_stores.create(
        name=vector_store_name,
        expires_after = vector_store_create_params.ExpiresAfter(
            anchor = "last_active_at",
            days = 3
        )
        )
    openai.beta.vector_stores.file_batches.create(
        vector_store_id=vector_store.id,
        file_ids=file_ids,
    )
    return vector_store.id


def poll_run_completion(
    run_id: str,
    thread_id: str,
    functions : Dict[str, Callable]
    ) -> bool:
    """
    Polls until the run is completed, failed, or cancelled, with a retry mechanism.

    Args:
        run_id (str): The ID of the run to monitor.
        thread_id (str): The thread ID associated with the run.

    Returns:
        bool: True if the run completed successfully, False otherwise.
    """
    retries = 0

    while retries <= MAX_RETRIES:
        while True:
            try:
                current_run = openai.beta.threads.runs.retrieve(
                    run_id=run_id,
                    thread_id=thread_id
                )
                status = current_run.status
                # Log status for debugging
                logging.debug(f"Run {run_id} current status: {status}")
                
                if status == 'completed':
                    logging.debug(f"Run {run_id} completed.")
                    # Ensure the thread has at least one assistant message
                    last_message = retrieve_last_assistant_message(thread_id)
                    if last_message:
                        return True
                    logging.error("Run completed, but no assistant response available.")
                    return False
                elif status in ['failed', 'expired', 'cancelled']:
                    logging.error(f"Run {run_id} ended with status: {status}")
                    logging.error(f"Details : {current_run.last_error}")
                    break  # Exit the inner loop to retry
                else:
                    logging.debug(f"Run {run_id} still in progress. Waiting...")
                    if status == "requires_action":
                        for tool_call in current_run.required_action.submit_tool_outputs.tool_calls:
                            if tool_call.function.name == "write_file_with_new_docstring":
                                return_value = functions[tool_call.function.name](**json.loads(tool_call.function.arguments))

                                if return_value:
                                    openai.beta.threads.runs.submit_tool_outputs(
                                        thread_id=thread_id,
                                        run_id=run_id,
                                        tool_outputs=[
                                            {
                                                "tool_call_id": tool_call.id,
                                                "output": ""
                                            }
                                        ]
                                    )
                                    return return_value
                                    
                        logging.error(f"Tool returned value : {return_value}")
                        raise Exception(f"Tool returned value : {return_value}")
                        
                    time.sleep(RETRY_BACKOFF)
            except Exception as e:
                logging.error(f"An error occurred while polling the run: {e}")
                break  # Exit the inner loop to retry
        
        retries += 1
        if retries > MAX_RETRIES:
            logging.error(f"Maximum retries reached for run {run_id}. Aborting.")
            return False
        
        logging.info(f"Retrying run {run_id} (attempt {retries}/{MAX_RETRIES})...")
        time.sleep(RETRY_BACKOFF * retries)  # Exponential backoff

    return False


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



    if hasattr(thread_messages[-1],'role') : 
        logging.debug(f"##### Success : ")
        logging.debug(f"role:{thread_messages[-1].role}")
        logging.debug(f"create_at:{thread_messages[-1].created_at}")
        logging.debug(f"status:{thread_messages[-1].status}\n")
    else :
        logging.error(f"##### Failure : ")
        logging.error(thread_messages)

    # try:
    #     extract_code_from_message(thread_messages[-1].content[-1].text.value)
    # except Exception:
    #     print("Error : last 5 messsages ")
    #     for message in thread_messages[-5]:
    #         print(f"#######################")
    #         print(f"role:{message.role}")
    #         print(f"create_at:{message.created_at}")
    #         print(f"status:{message.status}\n")
    return thread_messages[-1].content

