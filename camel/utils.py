# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import inspect
import os
import re
import time
import zipfile
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    cast,
)

import requests
import tiktoken

from camel.messages import OpenAIMessage
from camel.camel_typing import ModelType, TaskType

F = TypeVar('F', bound=Callable[..., Any])

from dotenv import load_dotenv
import os
import logging

# Load the .env file
load_dotenv()

# Retrieve the API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
server_url = os.getenv('TEXTGEN_API_URL')

def fetch_model_type_from_server(server_url):
    """Fetch the model type from the server.

    Args:
        server_url (str): The server URL where the model type information is hosted.

    Returns:
        str: The model type if fetched successfully, otherwise None.
    """
    response = requests.get(server_url)
    if response.status_code == 200:
        model_type = response.json()['result']
        logging.debug(f"Fetched model type from server: {model_type}")
        return model_type
    else:
        logging.error(f"Failed to fetch model type from server. Status code: {response.status_code}")

def count_tokens_openai_chat_models(
        messages: List[OpenAIMessage],
        encoding: Any,
) -> int:
    r"""Counts the number of tokens required to generate an OpenAI chat based
    on a given list of messages.

    Args:
        messages (List[OpenAIMessage]): The list of messages.
        encoding (Any): The encoding method to use.

    Returns:
        int: The number of tokens required.
    """
    num_tokens = 0
    for message in messages:
        # message follows <im_start>{role/name}\n{content}<im_end>\n
        num_tokens += 4
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens


def num_tokens_from_messages(
        messages: List[OpenAIMessage],
        model: ModelType,
) -> int:
    r"""Returns the number of tokens used by a list of messages.

    Args:
        messages (List[OpenAIMessage]): The list of messages to count the
            number of tokens for.
        model (ModelType): The OpenAI model used to encode the messages.

    Returns:
        int: The total number of tokens used by the messages.

    Raises:
        NotImplementedError: If the specified `model` is not implemented.

    References:
        - https://github.com/openai/openai-python/blob/main/chatml.md
        - https://platform.openai.com/docs/models/gpt-4
        - https://platform.openai.com/docs/models/gpt-3-5
    """
    try:
        value_for_tiktoken = model.value_for_tiktoken
        encoding = tiktoken.encoding_for_model(value_for_tiktoken)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    if model.is_openai or model.is_open_source:  # Updated to handle all models
        return count_tokens_openai_chat_models(messages, encoding)
    else:
        raise NotImplementedError(
            f"`num_tokens_from_messages`` is not presently implemented "
            f"for model {model}. "
            f"See https://github.com/openai/openai-python/blob/main/chatml.md "
            f"for information on how messages are converted to tokens. "
            f"See https://platform.openai.com/docs/models/gpt-4"
            f"or https://platform.openai.com/docs/models/gpt-3-5"
            f"for information about openai chat models.")


def get_model_token_limit(model: ModelType) -> int:
    r"""Returns the maximum token limit for a given model.

    Args:
        model (ModelType): The type of the model.

    Returns:
        int: The maximum token limit for the given model.
    """
    return model.token_limit  # Updated to use the token_limit property from your updated ModelType enum

def get_selected_model():
    """Fetch the selected model and its token limit.

    Returns:
        tuple: A tuple containing the selected model and its token limit. 
               Returns (None, 2048) if the model is not found.
    """
    model_name = fetch_model_type_from_server(server_url)
    
    model_type = None
    max_tokens = None

    for model in ModelType:
        if model.value == model_name:
            model_type = model.name
            max_tokens = get_model_token_limit(model)
            break

    if model_type and max_tokens:
        logging.debug(f"Found Model Type: {model_type}")
        logging.debug(f"Max Tokens: {max_tokens}")
        return model_type, max_tokens
    else:
        logging.warning("Model not found.")
        return 'GPT_3_5_TURBO', 2048
    

def openai_api_key_required(func: F) -> F:
    r"""Decorator that checks if the OpenAI API key is available in the
    environment variables.

    Args:
        func (callable): The function to be wrapped.

    Returns:
        callable: The decorated function.

    Raises:
        ValueError: If the OpenAI API key is not found in the environment
            variables.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        from camel.agents.chat_agent import ChatAgent
        if not isinstance(self, ChatAgent):
            raise ValueError("Expected ChatAgent")
        if self.model == ModelType.STUB:
            return func(self, *args, **kwargs)
        elif 'OPENAI_API_KEY' in os.environ:
            return func(self, *args, **kwargs)
        else:
            raise ValueError('OpenAI API key not found.')

    return wrapper


def print_text_animated(text, delay: float = 0.005, end: str = ""):
    r"""Prints the given text with an animated effect.

    Args:
        text (str): The text to print.
        delay (float, optional): The delay between each character printed.
            (default: :obj:`0.02`)
        end (str, optional): The end character to print after the text.
            (default: :obj:`""`)
    """
    for char in text:
        print(char, end=end, flush=True)
        time.sleep(delay)
    print('\n')


def get_prompt_template_key_words(template: str) -> Set[str]:
    r"""Given a string template containing curly braces {}, return a set of
    the words inside the braces.

    Args:
        template (str): A string containing curly braces.

    Returns:
        List[str]: A list of the words inside the curly braces.

    Example:
        >>> get_prompt_template_key_words('Hi, {name}! How are you {status}?')
        {'name', 'status'}
    """
    return set(re.findall(r'{([^}]*)}', template))


def get_first_int(string: str) -> Optional[int]:
    r"""Returns the first integer number found in the given string.

    If no integer number is found, returns None.

    Args:
        string (str): The input string.

    Returns:
        int or None: The first integer number found in the string, or None if
            no integer number is found.
    """
    match = re.search(r'\d+', string)
    if match:
        return int(match.group())
    else:
        return None


def download_tasks(task: TaskType, folder_path: str) -> None:
    # Define the path to save the zip file
    zip_file_path = os.path.join(folder_path, "tasks.zip")

    # Download the zip file from the Google Drive link
    response = requests.get("https://huggingface.co/datasets/camel-ai/"
                            f"metadata/resolve/main/{task.value}_tasks.zip")

    # Save the zip file
    with open(zip_file_path, "wb") as f:
        f.write(response.content)

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(folder_path)

    # Delete the zip file
    os.remove(zip_file_path)

def parse_doc(func: Callable) -> Dict[str, Any]:
    r"""Parse the docstrings of a function to extract the function name,
    description and parameters.

    Args:
        func (Callable): The function to be parsed.
    Returns:
        Dict[str, Any]: A dictionary with the function's name,
            description, and parameters.
    """

    doc = inspect.getdoc(func)
    if not doc:
        raise ValueError(
            f"Invalid function {func.__name__}: no docstring provided.")

    properties = {}
    required = []

    parts = re.split(r'\n\s*\n', doc)
    func_desc = parts[0].strip()

    args_section = next((p for p in parts if 'Args:' in p), None)
    if args_section:
        args_descs: List[Tuple[str, str, str, ]] = re.findall(
            r'(\w+)\s*\((\w+)\):\s*(.*)', args_section)
        properties = {
            name.strip(): {
                'type': type,
                'description': desc
            }
            for name, type, desc in args_descs
        }
        for name in properties:
            required.append(name)

    # Parameters from the function signature
    sign_params = list(inspect.signature(func).parameters.keys())
    if len(sign_params) != len(required):
        raise ValueError(
            f"Number of parameters in function signature ({len(sign_params)})"
            f" does not match that in docstring ({len(required)}).")

    for param in sign_params:
        if param not in required:
            raise ValueError(f"Parameter '{param}' in function signature"
                             " is missing in the docstring.")

    parameters = {
        "type": "object",
        "properties": properties,
        "required": required,
    }

    # Construct the function dictionary
    function_dict = {
        "name": func.__name__,
        "description": func_desc,
        "parameters": parameters,
    }

    return function_dict


def get_task_list(task_response: str) -> List[str]:
    r"""Parse the response of the Agent and return task list.

    Args:
        task_response (str): The string response of the Agent.

    Returns:
        List[str]: A list of the string tasks.
    """

    new_tasks_list = []
    task_string_list = task_response.strip().split('\n')
    # each task starts with #.
    for task_string in task_string_list:
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = ''.join(s for s in task_parts[0] if s.isnumeric())
            task_name = re.sub(r'[^\w\s_]+', '', task_parts[1]).strip()
            if task_name.strip() and task_id.isnumeric():
                new_tasks_list.append(task_name)
    return new_tasks_list