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
import re
from enum import Enum


class TaskType(Enum):
    AI_SOCIETY = "ai_society"
    CODE = "code"
    MISALIGNMENT = "misalignment"
    TRANSLATION = "translation"
    EVALUATION = "evaluation"
    SOLUTION_EXTRACTION = "solution_extraction"
    CHATDEV = "chat_dev"
    ROLE_DESCRIPTION = "role_description"
    DEFAULT = "default"


class RoleType(Enum):
    ASSISTANT = "assistant"
    USER = "user"
    CRITIC = "critic"
    EMBODIMENT = "embodiment"
    DEFAULT = "default"
    CHATDEV = "AgentTech"
    CHATDEV_COUNSELOR = "counselor"
    CHATDEV_CEO = "chief executive officer (CEO)"
    CHATDEV_CHRO = "chief human resource officer (CHRO)"
    CHATDEV_CPO = "chief product officer (CPO)"
    CHATDEV_CTO = "chief technology officer (CTO)"
    CHATDEV_PROGRAMMER = "programmer"
    CHATDEV_REVIEWER = "code reviewer"
    CHATDEV_TESTER = "software test engineer"
    CHATDEV_CCO = "chief creative officer (CCO)"


from enum import Enum

class ModelType(Enum):
    # OpenAI Models
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_3_5_TURBO_16K = "gpt-3.5-turbo-16k"
    GPT_4 = "gpt-4"
    GPT_4_32K = "gpt-4-32k"
    STUB = "stub"
    
    # Existing Open Source Models
    LLAMA_2 = "llama-2"
    VICUNA = "vicuna"
    VICUNA_16K = "vicuna-16k"
    
    # New Open Source Models
    WIZARDCODER_PYTHON_34B = "TheBloke_WizardCoder-Python-34B-V1.0-GPTQ"
    WIZARDLM_13B = "4bit_WizardLM-13B-Uncensored-4bit-128g"
    LLAMA_2_13B_CHAT_GERMAN = "jphme_Llama-2-13b-chat-german-GGML"
    OPENORCA_PLATYPUS2_13B = "Open-Orca_OpenOrca-Platypus2-13B"
    OPENORCA_OPENORCAXOPENCHAT = "Open-Orca_OpenOrcaxOpenChat-Preview2-13B"
    CODELLAMA_34B = "TheBloke_CodeLlama-34B-Instruct-GPTQ"
    LLAMA_2_13B_CHAT = "TheBloke_Llama-2-13B-chat-GPTQ"
    LLAMA_2_13B_GERMAN = "TheBloke_llama-2-13B-German-Assistant-v2-GPTQ"
    LUNA_AI_LLAMA2 = "TheBloke_Luna-AI-Llama2-Uncensored-GPTQ"
    STARCODERPLUS_GUANACO = "TheBloke_Starcoderplus-Guanaco-GPT4-15B-V1.0-GPTQ"
    WIZARD_VICUNA_13B = "TheBloke_Wizard-Vicuna-13B-Uncensored-SuperHOT-8K-GPTQ"
    YARN_LLAMA_2_13B = "TheBloke_Yarn-Llama-2-13B-128K-GPTQ"

    @property
    def value_for_tiktoken(self) -> str:
        return self.value if self.name != "STUB" else "gpt-3.5-turbo"

    @property
    def is_openai(self) -> bool:
        if self.name in {
                "GPT_3_5_TURBO",
                "GPT_3_5_TURBO_16K",
                "GPT_4",
                "GPT_4_32K",
        }:
            return True
        else:
            return False

    @property
    def is_open_source(self) -> bool:
        if self.name in {
                "LLAMA_2",
                "VICUNA",
                "VICUNA_16K",
                "WIZARDCODER_PYTHON_34B",
                "WIZARDLM_13B",
                "LLAMA_2_13B_CHAT_GERMAN",
                "OPENORCA_PLATYPUS2_13B",
                "OPENORCA_OPENORCAXOPENCHAT",
                "CODELLAMA_34B",
                "LLAMA_2_13B_CHAT",
                "LLAMA_2_13B_GERMAN",
                "LUNA_AI_LLAMA2",
                "STARCODERPLUS_GUANACO",
                "WIZARD_VICUNA_13B",
                "YARN_LLAMA_2_13B"
        }:
            return True
        else:
            return False

    @property
    def token_limit(self) -> int:
        if self is ModelType.GPT_3_5_TURBO:
            return 4096
        elif self is ModelType.GPT_3_5_TURBO_16K:
            return 16384
        elif self is ModelType.GPT_4:
            return 8192
        elif self is ModelType.GPT_4_32K:
            return 32768
        elif self is ModelType.STUB:
            return 4096
        elif self is ModelType.LLAMA_2:
            return 4096
        elif self is ModelType.VICUNA:
            return 2048
        elif self is ModelType.VICUNA_16K:
            return 16384
        elif self is ModelType.WIZARDCODER_PYTHON_34B:
            return 16384
        elif self is ModelType.WIZARDLM_13B:
            return 2048
        elif self is ModelType.OPENORCA_PLATYPUS2_13B:
            return 4096
        elif self is ModelType.OPENORCA_OPENORCAXOPENCHAT:
            return 4096
        elif self is ModelType.CODELLAMA_34B:
            return 16384
        elif self is ModelType.LLAMA_2_13B_CHAT:
            return 2048
        elif self is ModelType.LLAMA_2_13B_GERMAN:
            return 2048
        elif self is ModelType.LUNA_AI_LLAMA2:
            return 2048
        elif self is ModelType.STARCODERPLUS_GUANACO:
            return 8192
        elif self is ModelType.WIZARD_VICUNA_13B:
            return 8192
        elif self is ModelType.YARN_LLAMA_2_13B:
            return 131072
        else:
            raise ValueError("Unknown model type")

def validate_model_name(self, model_name: str) -> bool:
    if self is ModelType.VICUNA:
        pattern = r'^vicuna-\d+b-v\d+\.\d+$'
        return bool(re.match(pattern, model_name))
    elif self is ModelType.VICUNA_16K:
        pattern = r'^vicuna-\d+b-v\d+\.\d+-16k$'
        return bool(re.match(pattern, model_name))
    elif self is ModelType.LLAMA_2:
        return "llama2" in model_name.lower() or "llama-2" in model_name.lower()
    elif self is ModelType.WIZARDCODER_PYTHON_34B:
        return "wizardcoder" in model_name.lower() and "python" in model_name.lower()
    elif self is ModelType.WIZARDLM_13B:
        return "wizardlm" in model_name.lower() and "13b" in model_name.lower()
    elif self is ModelType.LLAMA_2_13B_CHAT_GERMAN:
        return "llama-2-13b-chat-german" in model_name.lower()
    elif self is ModelType.OPENORCA_PLATYPUS2_13B:
        return "openorca-platypus2-13b" in model_name.lower()
    elif self is ModelType.OPENORCA_OPENORCAXOPENCHAT:
        return "openorca-openorcaxopenchat" in model_name.lower()
    elif self is ModelType.CODELLAMA_34B:
        return "codellama-34b" in model_name.lower()
    elif self is ModelType.LLAMA_2_13B_CHAT:
        return "llama-2-13b-chat" in model_name.lower()
    elif self is ModelType.LLAMA_2_13B_GERMAN:
        return "llama-2-13b-german" in model_name.lower()
    elif self is ModelType.LUNA_AI_LLAMA2:
        return "luna-ai-llama2" in model_name.lower()
    elif self is ModelType.STARCODERPLUS_GUANACO:
        return "starcoderplus-guanaco" in model_name.lower()
    elif self is ModelType.WIZARD_VICUNA_13B:
        return "wizard-vicuna-13b" in model_name.lower()
    elif self is ModelType.YARN_LLAMA_2_13B:
        return "yarn-llama-2-13b" in model_name.lower()
    else:
        return self.value.replace("-", "_").lower() in model_name.replace("-", "_").lower()

class PhaseType(Enum):
    REFLECTION = "reflection"
    RECRUITING_CHRO = "recruiting CHRO"
    RECRUITING_CPO = "recruiting CPO"
    RECRUITING_CTO = "recruiting CTO"
    DEMAND_ANALYSIS = "demand analysis"
    BRAINSTORMING = "brainstorming"
    CHOOSING_LANGUAGE = "choosing language"
    RECRUITING_PROGRAMMER = "recruiting programmer"
    RECRUITING_REVIEWER = "recruiting reviewer"
    RECRUITING_TESTER = "recruiting software test engineer"
    RECRUITING_CCO = "recruiting chief creative officer"
    CODING = "coding"
    CODING_COMPLETION = "coding completion"
    CODING_AUTOMODE = "coding auto mode"
    REVIEWING_COMMENT = "review comment"
    REVIEWING_MODIFICATION = "code modification after reviewing"
    ERROR_SUMMARY = "error summary"
    MODIFICATION = "code modification"
    ART_ELEMENT_ABSTRACTION = "art element abstraction"
    ART_ELEMENT_INTEGRATION = "art element integration"
    CREATING_ENVIRONMENT_DOCUMENT = "environment document"
    CREATING_USER_MANUAL = "user manual"


__all__ = ["TaskType", "RoleType", "ModelType", "PhaseType"]
