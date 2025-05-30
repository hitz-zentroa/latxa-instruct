"""
api.py
========

This module provides an abstraction layer for integrating various AI models 
(OpenAI, Google Generative AI, Cohere, Anthropic, and vLLM) into the Latxa 
assistant. It defines a common interface for handling chat-based interactions 
and generating responses using different AI APIs.


License:
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at:

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and 
    limitations under the License.

Author:
    Oscar Sainz (oscar.sainz@ehu.eus)
"""
from dataclasses import dataclass
import time
from typing import Any, Dict, Generator, List, Literal, Tuple
import os

from copy import deepcopy


from openai import OpenAI
import google.generativeai as genai
from google.generativeai.types.generation_types import GenerationConfig as GenConfig
import cohere
import anthropic

system_prompt = (
    "You are a helpful Artificial Intelligence assistant called Latxa, "
    "created and developed by HiTZ, the Basque Center for Language Technology research center. "
    "The user will engage in a multi-round conversation with you, asking "
    "initial questions and following up with additional related questions. "
    "Your goal is to provide thorough, relevant and insightful responses "
    "to help the user with their queries. Every conversation will be "
    "conducted in standard Basque, this is, the first question from the user will be "
    "in Basque, and you should respond in formal Basque as well. Conversations will "
    "cover a wide range of topics, including but not limited to general "
    "knowledge, science, technology, entertainment, coding, mathematics, "
    "and more. Today is {date}."
)


def today():
    return time.strftime("%A %B %e, %Y", time.gmtime())


@dataclass
class GenerationConfig:
    max_tokens: int
    temperature: float
    top_p: float
    stop: str | List[str] = None
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0


class API:
    SYSTEM_ROLE = "system"
    USER_ROLE = "user"
    MODEL_ROLE = "assistant"

    def __init__(
        self,
        model_name: str,
        url: str = None,
        api_key: str = None,
        sysprompt_format: Literal["system", "no", "user-turn"] = "system",
        generation_config: GenerationConfig = None,
    ):
        self.url = url
        self.model_name = model_name
        self.api_key = api_key
        self.sysprompt_format = sysprompt_format
        self.generation_config = generation_config

    def get_chat_stream(
        self, history: List[Tuple[str, str]], **kwargs
    ) -> Generator[List[Tuple[str, str]], None, None]:
        raise NotImplementedError

    def prepare_generation_config_for_api(
        self, generation_config: GenerationConfig
    ) -> Any:
        return deepcopy(generation_config.__dict__)

    def format_history(
        self, history: List[Dict[str, str]], add_system_prompt: bool = True
    ) -> List[Dict[str, str]]:
        formatted_history = history.copy()
        if add_system_prompt and self.sysprompt_format != "no":
            if self.sysprompt_format == "system":
                formatted_history = [
                    {
                        "role": self.SYSTEM_ROLE,
                        "content": system_prompt.format(date=today()),
                    }
                ] + formatted_history
            elif self.sysprompt_format == "user-turn":
                formatted_history = [
                    {"role": "user", "content": system_prompt.format(date=today())},
                    {
                        "role": "assistant",
                        "content": "Ados! Prest nago zuri laguntzeko. Esaidazu, nola lagundu dezaket?",
                    },
                    *formatted_history,
                ]
            else:
                ValueError(
                    "System prompt format must be: 'system', 'user-turn' or 'no'"
                )

        # Remove unnecesary data
        for i, message in enumerate(formatted_history):
            formatted_history[i] = {
                "role": message["role"],
                "content": message["content"],
            }

        return formatted_history


class OpenAIAPI(API):
    def __init__(
        self,
        model_name: str,
        url: str = None,
        api_key: str = None,
        sysprompt_format: Literal["system", "no", "user-turn"] = "system",
        generation_config: GenerationConfig = None,
    ):
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY", None)

        self.endpoint = OpenAI(base_url=url, api_key=api_key)
        super().__init__(
            url=url,
            model_name=model_name,
            api_key=api_key,
            sysprompt_format=sysprompt_format,
            generation_config=generation_config,
        )

    def prepare_generation_config_for_api(
        self, generation_config: GenerationConfig
    ) -> Any:
        _dict = deepcopy(generation_config.__dict__)

        _dict.pop("stop")
        _dict.pop("repetition_penalty")
        max_tokens = _dict.pop("max_tokens")
        _dict["max_completion_tokens"] = max_tokens

        return _dict

    def get_chat_stream(
        self, history: List[Tuple[str, str]], **kwargs
    ) -> Generator[List[Tuple[str, str]], None, None]:
        history = deepcopy(history)

        formatted_history = self.format_history(history)

        history.append({"role": "assistant", "content": ""})
        try:
            stream = self.endpoint.chat.completions.create(
                model=self.model_name,
                messages=formatted_history,
                stream=True,
                **self.prepare_generation_config_for_api(self.generation_config),
            )

            for chunk in stream:
                new_token = chunk.choices[0].delta.content
                if new_token in ["<|eot_id|>"]:
                    print(f"Found <|eot_id|> in output. Model: {self.model_name}")
                    return history
                    continue
                if new_token is not None:
                    history[-1]["content"] += new_token

                    if history[-1]["content"].endswith("]]>"):
                        history[-1]["content"] = history[-1]["content"][:-3]
                        return history
                    yield history
        except Exception as e:
            print(e)
            history[-1]["content"] = (
                "API-arekin arazoak egon dira. Barkatu eragozpenak."
            )
            yield history


class vLLMAPI(OpenAIAPI):
    def __init__(
        self,
        model_name: str,
        url: str,
        api_key: str = None,
        sysprompt_format: Literal["system", "no", "user-turn"] = "system",
        generation_config: GenerationConfig = None,
    ):
        if not api_key:
            api_key = "EMPTY"

        super().__init__(
            url=url,
            model_name=model_name,
            api_key=api_key,
            sysprompt_format=sysprompt_format,
            generation_config=generation_config,
        )

    def prepare_generation_config_for_api(
        self, generation_config: GenerationConfig
    ) -> Any:
        _dict = deepcopy(generation_config.__dict__)

        max_tokens = _dict.pop("max_tokens")
        stop = _dict.pop("stop")
        repetition_penalty = _dict.pop("repetition_penalty")
        _dict["extra_body"] = {
            "max_tokens": max_tokens,
            "stop": stop,
            "repetition_penalty": repetition_penalty,
        }

        return _dict


class GoogleAPI(API):
    USER_ROLE = "user"
    MODEL_ROLE = "model"

    def __init__(
        self,
        model_name: str = "models/gemini-1.5-flash",
        api_key: str = None,
        generation_config: GenerationConfig = None,
    ):
        if not api_key:
            api_key = os.environ.get("GOOGLE_API_KEY", None)

        genai.configure(api_key=api_key)
        self.endpoint = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt.format(date=today()),
            generation_config=GenConfig(
                max_output_tokens=generation_config.max_tokens,
                temperature=generation_config.temperature,
                top_p=generation_config.top_p,
                frequency_penalty=generation_config.frequency_penalty
                if model_name != "models/gemini-1.5-flash"
                else None,
            ),
        )
        super().__init__(
            model_name=model_name, api_key=api_key, generation_config=generation_config
        )

    def format_history(self, history: List[Tuple[str, str]]) -> List[Dict[str, str]]:
        formatted_history = []

        for message in history:
            role = self.USER_ROLE if message["role"] == "user" else self.MODEL_ROLE
            formatted_history.append({"role": role, "parts": [message["content"]]})

        return formatted_history

    def get_chat_stream(
        self, history: List[Tuple[str, str]], **kwargs
    ) -> Generator[List[Tuple[str, str]], None, None]:
        history = deepcopy(history)

        formatted_history = self.format_history(history)
        history.append({"role": "assistant", "content": ""})

        try:
            stream = self.endpoint.generate_content(formatted_history, stream=True)
            for chunk in stream:
                if chunk.candidates and chunk.parts:
                    new_token = chunk.text

                    if new_token is not None:
                        # history[-1] = (message, history[-1][1] + new_token)
                        history[-1]["content"] += new_token
                        yield history

        except Exception as e:
            print(type(e), e)
            # Finish reason 2 means the MAX_TOKENS limit was reached
            if chunk.candidates[0].finish_reason not in [2, 3]:
                history[-1]["content"] = (
                    "API-arekin arazoak egon dira. Barkatu eragozpenak."
                )
            elif chunk.candidates[0].finish_reason == 3:
                history[-1]["content"] = (
                    "Hizkuntza-eredu bezela ezin dut galdera horri erantzun."
                )

            yield history


class CohereAPI(API):
    def __init__(
        self,
        model_name: str = "command-r-plus-08-2024",
        api_key: str = None,
        generation_config: GenerationConfig = None,
    ):
        if not api_key:
            api_key = os.environ.get("COHERE_API_KEY", None)

        self.endpoint = cohere.ClientV2(api_key=api_key)
        super().__init__(
            model_name=model_name, api_key=api_key, generation_config=generation_config
        )

    def prepare_generation_config_for_api(
        self, generation_config: GenerationConfig
    ) -> Any:
        _dict = deepcopy(generation_config.__dict__)
        _dict.pop("stop")

        top_p = _dict.pop("top_p")
        _dict["p"] = top_p

        return _dict

    def get_chat_stream(
        self, history: List[Tuple[str, str]], **kwargs
    ) -> Generator[List[Tuple[str, str]], None, None]:
        history = deepcopy(history)

        formatted_history = self.format_history(history)

        stream = self.endpoint.chat_stream(
            model=self.model_name,
            messages=formatted_history,
            **self.prepare_generation_config_for_api(self.generation_config),
        )

        history.append({"role": "assistant", "content": ""})
        try:
            for chunk in stream:
                if chunk.type == "content-delta":
                    new_token = chunk.delta.message.content.text

                    if new_token in ["<|eot_id|>"]:
                        continue
                    if new_token is not None:
                        # history[-1] = (message, history[-1][1] + new_token)
                        history[-1]["content"] += new_token
                        yield history

        except Exception as e:
            print(e)
            history[-1]["content"] = (
                "API-arekin arazoak egon dira. Barkatu eragozpenak."
            )
            yield history


class AnthropicAPI(API):
    def __init__(
        self,
        model_name: str = "claude-3-haiku-20240307",
        api_key: str = None,
        generation_config: GenerationConfig = None,
    ):
        if not api_key:
            api_key = os.environ.get("ANTHROPIC_API_KEY", None)

        self.endpoint = anthropic.Anthropic(api_key=api_key)
        super().__init__(
            model_name=model_name, api_key=api_key, generation_config=generation_config
        )

    def prepare_generation_config_for_api(
        self, generation_config: GenerationConfig
    ) -> Any:
        _dict = deepcopy(generation_config.__dict__)

        _ = _dict.pop("frequency_penalty")
        _dict.pop("repetition_penalty")
        _dict.pop("stop")

        _dict["system"] = system_prompt.format(date=today())

        return _dict

    def get_chat_stream(
        self, history: List[Tuple[str, str]], **kwargs
    ) -> Generator[List[Tuple[str, str]], None, None]:
        history = deepcopy(history)

        formatted_history = self.format_history(history, add_system_prompt=False)

        history.append({"role": "assistant", "content": ""})
        try:
            with self.endpoint.messages.stream(
                model=self.model_name,
                messages=formatted_history,
                **self.prepare_generation_config_for_api(self.generation_config),
            ) as stream:
                # Start streaming
                for new_token in stream.text_stream:
                    if new_token is not None:
                        history[-1]["content"] += new_token
                        yield history

        except Exception as e:
            print(e)
            history[-1]["content"] = (
                "API-arekin arazoak egon dira. Barkatu eragozpenak."
            )
            yield history
