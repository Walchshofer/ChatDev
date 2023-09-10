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
from abc import ABC, abstractmethod
from typing import Any, Dict
import requests
import openai
import tiktoken
from transformers import AutoTokenizer
from dotenv import load_dotenv
import os

from camel.camel_typing import ModelType
from chatdev.utils import log_and_print_online
import logging
#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


class ModelBackend(ABC):
    @abstractmethod
    def run(self, *args, **kwargs) -> Dict[str, Any]:
        pass


class OpenAIModel(ModelBackend):
    def __init__(self, model_type: ModelType, model_config_dict: Dict) -> None:
        super().__init__()
        self.model_type = model_type
        self.model_config_dict = model_config_dict

        if self.model_type.is_openai:
            self.tokenizer = None
        elif self.model_type.is_open_source:
            server_url = "http://localhost:5000/api/v1/model"
            model_name = self.fetch_model_name_from_server(server_url)
            model_base_path = self.get_model_base_path()
            full_model_path = os.path.join(model_base_path, model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(full_model_path)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    @staticmethod
    def fetch_model_name_from_server(server_url):
        response = requests.get(server_url)
        if response.status_code == 200:
            model_name = response.json()['result']
            logging.debug(f"Fetched model name from server: {model_name}")
            return model_name
        else:
            logging.error(f"Failed to fetch model name from server. Status code: {response.status_code}")
            return None


    @staticmethod
    def get_model_base_path():
        load_dotenv()
        return os.getenv("MODEL_PATH")

    def run(self, *args, **kwargs) -> Dict[str, Any]:
        string = "\n".join([message["content"] for message in kwargs["messages"]])

        if self.model_type.is_openai:
            encoding = tiktoken.encoding_for_model(self.model_type.value_for_tiktoken)
            num_prompt_tokens = len(encoding.encode(string))
        elif self.model_type.is_open_source:
            num_prompt_tokens = len(self.tokenizer(string, return_attention_mask=False)["input_ids"])

        gap_between_send_receive = 15 * len(kwargs["messages"])
        num_prompt_tokens += gap_between_send_receive

        num_max_token = self.model_type.token_limit
        # DEBUG: print the token limit to confirm
        print(f"model_backend DEBUG: num_max_token (token_limit from camel_typing): {num_max_token}")

        num_max_completion_tokens = num_max_token - num_prompt_tokens
        self.model_config_dict['max_tokens'] = num_max_completion_tokens
        
        #logging.debug(f"About to make an API call with model: {self.model_type} and max tokens: {num_max_completion_tokens}")

        response = openai.ChatCompletion.create(
            *args, **kwargs, model=self.model_type.value, **self.model_config_dict
        )

        log_and_print_online(
            "**[OpenAI_Usage_Info Receive]**\nprompt_tokens: {}\ncompletion_tokens: {}\ntotal_tokens: {}\n".format(
                response["usage"]["prompt_tokens"], response["usage"]["completion_tokens"],
                response["usage"]["total_tokens"]
            )
        )
        
        if not isinstance(response, Dict):
            raise RuntimeError("Unexpected return from OpenAI API")
        return response


class StubModel(ModelBackend):
    r"""A dummy model used for unit tests."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def run(self, *args, **kwargs) -> Dict[str, Any]:
        ARBITRARY_STRING = "Lorem Ipsum"

        return dict(
            id="stub_model_id",
            usage=dict(),
            choices=[
                dict(finish_reason="stop",
                     message=dict(content=ARBITRARY_STRING, role="assistant"))
            ],
        )


class ModelFactory:
    r"""Factory of backend models.

    Raises:
        ValueError: in case the provided model type is unknown.
    """

    @staticmethod
    def create(model_type: ModelType, model_config_dict: Dict) -> ModelBackend:
        default_model_type = ModelType.GPT_3_5_TURBO

        if model_type.is_openai or model_type.is_open_source or model_type is None:
            model_class = OpenAIModel
        elif model_type == ModelType.STUB:
            model_class = StubModel
        else:
            raise ValueError(f"Unknown model: {model_type}")

        if model_type is None:
            model_type = default_model_type

        inst = model_class(model_type, model_config_dict)
        return inst
