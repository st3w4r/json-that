import sys
import json
import os
import requests
from abc import ABC, abstractmethod
from typing import Dict, Type, Optional, List, Generator, Union
import yaml
import argparse
from enum import Enum
from .version import __version__


class ProviderType(Enum):
    CLOUD = "cloud"
    LOCAL = "local"


class LLMProvider(ABC):
    @abstractmethod
    def transform_text_to_json(
        self, text: str, schema: Optional[str] = None, stream: bool = False
    ) -> Union[dict[str, any], Generator[str, None, None]]:
        pass

    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        pass


class ProviderInfo:
    def __init__(self, provider_class: Type[LLMProvider], provider_type: ProviderType):
        self.provider_class = provider_class
        self.provider_type = provider_type


class ProviderRegistry:
    _providers: Dict[str, ProviderInfo] = {}

    @classmethod
    def register(cls, name: str, provider_type: ProviderType):
        def decorator(provider_class: Type[LLMProvider]):
            cls._providers[name] = ProviderInfo(provider_class, provider_type)
            return provider_class

        return decorator

    @classmethod
    def get_provider_info(cls, name: str) -> ProviderInfo:
        provider_info = cls._providers.get(name)
        if not provider_info:
            raise ValueError(f"Unsupported LLM provider: {name}")
        return provider_info

    @classmethod
    def get_available_providers(cls) -> List[str]:
        return list(cls._providers.keys())


@ProviderRegistry.register("openai", ProviderType.CLOUD)
class OpenAIProvider(LLMProvider):
    supports_streaming = True

    def __init__(self, api_key: str):
        self.api_key = api_key

    def _prepare_request_data(self, text: str, schema: Optional[str] = None) -> Dict:
        system_message = "Transform the raw text to JSON\n"
        if schema:
            system_message += (
                f"Use the following JSON schema to structure the output: {schema}\n"
            )
            system_message += (
                "Without including extra schema information back in the response."
            )

        return {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": f"Transform the following text into a JSON format: {text}",
                },
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def transform_text_to_json(
        self, text: str, schema: Optional[str] = None, stream: bool = False
    ) -> Union[dict[str, any], Generator[str, None, None]]:
        headers = self._get_headers()
        data = self._prepare_request_data(text, schema)
        data["stream"] = stream

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            stream=stream,
        )
        response.raise_for_status()

        if stream:
            return self._stream_response(response)
        else:
            result = response.json()
            transformed_text = result["choices"][0]["message"]["content"]
            return json.loads(transformed_text)

    def _stream_response(
        self, response: requests.Response
    ) -> Generator[str, None, None]:
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    line = line[6:]  # Remove 'data: ' prefix
                    if line.strip() == "[DONE]":
                        break
                    try:
                        json_response = json.loads(line)
                        content = json_response["choices"][0]["delta"].get(
                            "content", ""
                        )
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue


@ProviderRegistry.register("mistral", ProviderType.CLOUD)
class MistralProvider(LLMProvider):
    supports_streaming = True

    def __init__(self, api_key: str):
        self.api_key = api_key

    def _prepare_request_data(self, text: str, schema: Optional[str] = None) -> Dict:
        system_message = "Transform the raw text to JSON\n"
        if schema:
            system_message += (
                f"Use the following JSON schema to structure the output: {schema}\n"
            )
            system_message += (
                "Without including extra schema information back in the response."
            )

        return {
            "model": "mistral-large-latest",
            "messages": [
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": f"Transform the following text into a JSON format: {text}",
                },
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def transform_text_to_json(
        self, text: str, schema: Optional[str] = None, stream: bool = False
    ) -> Union[dict[str, any], Generator[str, None, None]]:
        headers = self._get_headers()
        data = self._prepare_request_data(text, schema)
        data["stream"] = stream

        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers=headers,
            json=data,
            stream=stream,
        )
        response.raise_for_status()

        if stream:
            return self._stream_response(response)
        else:
            result = response.json()
            transformed_text = result["choices"][0]["message"]["content"]
            return json.loads(transformed_text)

    def _stream_response(
        self, response: requests.Response
    ) -> Generator[str, None, None]:
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    line = line[6:]  # Remove 'data: ' prefix
                    if line.strip() == "[DONE]":
                        break
                    try:
                        json_response = json.loads(line)
                        content = json_response["choices"][0]["delta"].get(
                            "content", ""
                        )
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue


@ProviderRegistry.register("claude", ProviderType.CLOUD)
class ClaudeProvider(LLMProvider):
    supports_streaming = True

    def __init__(self, api_key: str):
        self.api_key = api_key

    def transform_text_to_json(
        self, text: str, schema: Optional[str] = None, stream: bool = False
    ) -> Union[dict[str, any], Generator[str, None, None]]:
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        system_message = "transform the raw text to json\n"
        if schema:
            system_message += (
                f" Use the following JSON schema to structure the output: {schema}\n"
            )
            system_message += (
                "Without including extra schema information back in the response."
            )

        full_user_message = (
            system_message
            + f"\nTransform the following text into a JSON format: {text}\n\nOnly output json response without any comment or extra information."
        )

        data = {
            "model": "claude-3-5-sonnet-20240620",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": full_user_message}],
            "temperature": 0,  # Set temperature to 0 for deterministic output,
            "stream": stream,
        }

        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=data,
            stream=stream,
        )
        response.raise_for_status()

        if stream:
            return self._stream_response(response)
        else:
            result = response.json()
            transformed_text = result["content"][0]["text"]
            return json.loads(transformed_text)

    def _stream_response(
        self, response: requests.Response
    ) -> Generator[str, None, None]:
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("event: content_block_stop") or line.startswith(
                    "event: message_stop"
                ):
                    break
                if line.startswith("data: "):
                    line = line[6:]  # Remove 'data: ' prefix
                    try:
                        json_response = json.loads(line)
                        type_msg = json_response.get("type")
                        if type_msg == "content_block_start":
                            content = json_response.get("ccontent_block", None)
                            if content:
                                text = content.get("text", None)
                                if text:
                                    yield text
                        elif type_msg == "content_block_delta":
                            content = json_response.get("delta", None)
                            if content:
                                text = content.get("text", None)
                                if text:
                                    yield text
                    except json.JSONDecodeError:
                        continue


@ProviderRegistry.register("ollama", ProviderType.LOCAL)
class OllamaProvider(LLMProvider):
    supports_streaming = True

    def __init__(self, api_url: str = "http://127.0.0.1:11434", model: str = "llama3"):
        self.api_url = api_url
        self.model = model

    def transform_text_to_json(
        self, text: str, schema: Optional[str] = None, stream: bool = False
    ) -> Union[dict[str, any], Generator[str, None, None]]:
        headers = {
            "Content-Type": "application/json",
        }

        prompt = f"Transform the following text into a JSON format: {text}\n"
        if schema:
            prompt += (
                f"Use the following JSON schema to structure the output: {schema}\n"
            )
        prompt += "Only output json response without any comment or extra information."

        data = {
            "prompt": prompt,
            "model": self.model,
            "format": "json",
            "stream": stream,
            "options": {
                "temperature": 0,  # Set temperature to 0 for deterministic output
            },
        }

        response = requests.post(
            f"{self.api_url}/api/generate",
            headers=headers,
            json=data,
            stream=stream,
        )
        response.raise_for_status()

        if stream:
            return self._stream_response(response)
        else:
            result = response.json()
            transformed_text = result["response"]
            return json.loads(transformed_text)

    def _stream_response(
        self, response: requests.Response
    ) -> Generator[str, None, None]:
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                try:
                    json_response = json.loads(line)
                    if json_response.get("done", False):
                        break
                    response = json_response.get("response", None)
                    if response:
                        yield response
                except json.JSONDecodeError:
                    continue


def display_streaming_response(generator: Generator[str, None, None]):
    for chunk in generator:
        sys.stdout.write(chunk)
        sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()


class Config:
    def __init__(self):
        self.config_file = self.get_config_file_path()
        self.config = self.load_config()

    def get_config_file_path(self):
        config_dir = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
        return os.path.join(config_dir, "jsonthat", "config.yaml")

    def load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, "r") as f:
                return yaml.safe_load(f)
        return {"providers": {}, "default_provider": None}

    def save_config(self):
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        with open(self.config_file, "w") as f:
            yaml.dump(self.config, f)

    def get_provider_config(self, provider_name: str) -> Dict[str, str]:
        return self.config["providers"].get(provider_name, {})

    def set_provider_config(self, provider_name: str, config: Dict[str, str]):
        if "providers" not in self.config:
            self.config["providers"] = {}
        self.config["providers"][provider_name] = config
        self.save_config()

    def get_default_provider(self) -> Optional[str]:
        return self.config.get("default_provider")

    def set_default_provider(self, provider_name: str):
        self.config["default_provider"] = provider_name
        self.save_config()

    def display_config(self):
        print(f"Config file path: {self.config_file}")

        if not os.path.exists(self.config_file):
            print(
                "\nConfig file does not exist. Run 'jt --setup' to create a configuration."
            )
            return

        print("\nCurrent configuration:")
        if not self.config or (
            not self.config.get("providers") and not self.config.get("default_provider")
        ):
            print("Configuration is empty. Run 'jt --setup' to configure the tool.")
        else:
            print(yaml.dump(self.config, default_flow_style=False))


def get_provider(config: Config, provider_name: Optional[str] = None) -> LLMProvider:
    if not provider_name:
        provider_name = (
            os.environ.get("LLM_PROVIDER") or config.get_default_provider() or "openai"
        )

    provider_info = ProviderRegistry.get_provider_info(provider_name)
    provider_config = config.get_provider_config(provider_name)

    if provider_info.provider_type == ProviderType.CLOUD:
        api_key = os.environ.get("LLM_API_KEY") or provider_config.get("api_key")
        if not api_key:
            raise ValueError(
                f"API key not found for {provider_name}.\nPlease run the setup command:\njt --setup\n"
            )
        provider_config = {"api_key": api_key}
    elif provider_name == "ollama":
        api_url = os.environ.get("OLLAMA_API_URL") or provider_config.get(
            "api_url", "http://127.0.0.1:11434"
        )
        model = os.environ.get("OLLAMA_MODEL") or provider_config.get("model", "llama3")
        provider_config = {"api_url": api_url, "model": model}

    return provider_info.provider_class(**provider_config)


def read_stdin() -> str:
    return sys.stdin.read().strip()


def read_schema_file(file_path: str) -> str:
    try:
        with open(file_path, "r") as f:
            return f.read().strip()
    except IOError as e:
        print(f"Error reading schema file: {e}", file=sys.stderr)
        sys.exit(1)


def setup_command(config: Config):
    print("Welcome to 'json that' CLI setup!")

    available_providers = ProviderRegistry.get_available_providers()
    provider_list = "/".join(available_providers)

    while True:
        try:
            provider = input(f"Choose your LLM provider ({provider_list}): ").lower()
            if provider in available_providers:
                break
            print(
                f"Invalid choice. Please enter one of: {', '.join(available_providers)}"
            )
        except KeyboardInterrupt:
            print("\nSetup aborted.")
            sys.exit(1)
        except EOFError:
            print("\nSetup aborted.")
            sys.exit(1)

    provider_info = ProviderRegistry.get_provider_info(provider)
    provider_config = {}

    if provider_info.provider_type == ProviderType.CLOUD:
        try:
            api_key = input(f"Enter your {provider.capitalize()} API key: ")
            provider_config["api_key"] = api_key
        except KeyboardInterrupt:
            print("\nSetup aborted.")
            sys.exit(1)
        except EOFError:
            print("\nSetup aborted.")
            sys.exit(1)
    elif provider == "ollama":
        try:
            api_url = input("Enter Ollama API URL (default: http://127.0.0.1:11434): ")
            if not api_url:
                api_url = "http://127.0.0.1:11434"
            provider_config["api_url"] = api_url

            model = input("Enter Ollama model name (default: llama3): ")
            if not model:
                model = "llama3"
            provider_config["model"] = model
        except KeyboardInterrupt:
            print("\nSetup aborted.")
            sys.exit(1)
        except EOFError:
            print("\nSetup aborted.")
            sys.exit(1)

    config.set_provider_config(provider, provider_config)

    set_default = input(
        "Do you want to set this as the default provider? (y/n): "
    ).lower()
    if set_default == "y":
        config.set_default_provider(provider)

    print(f"Configuration saved to {config.config_file}")


def example():
    print()
    print("examples:")
    print("  echo 'raw text' | jsonthat")
    print("  echo 'raw text' | jt")
    print("  echo 'raw text' | jt --schema schema.json")
    print("  echo 'raw text' | jt --provider openai")
    print("  echo 'raw text' | jt --stream")
    print("""
  echo 'my name is jay' | jt
  {
    "name": "Jay"
  }""")
    print()


class CustomHelpParser(argparse.ArgumentParser):
    def print_help(self):
        super().print_help()
        example()


def main():
    config = Config()

    parser = CustomHelpParser(description="Text to JSON CLI")
    parser.add_argument("--setup", action="store_true", help="Run the setup command")
    parser.add_argument("--schema", type=str, help="Path to the schema file")
    parser.add_argument("--provider", type=str, help="Specify the LLM provider to use")
    parser.add_argument(
        "--version", action="store_true", help="Show the version and exit"
    )
    parser.add_argument(
        "--config", action="store_true", help="Display current configuration"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the output",
    )
    args = parser.parse_args()

    if args.version:
        print(f"jsonthat CLI version {__version__}")
        return

    if args.setup:
        setup_command(config)
        return

    if args.config:
        config.display_config()
        return

    if sys.stdin.isatty():
        parser.print_help()
        return

    try:
        provider = get_provider(config, args.provider)
    except ValueError as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

    input_text = read_stdin()
    if not input_text:
        print("Error: No input provided.", file=sys.stderr)
        sys.exit(1)

    schema = None
    if args.schema:
        schema = read_schema_file(args.schema)

    try:
        if args.stream and not provider.supports_streaming:
            print(
                f"Warning: {args.provider} does not support streaming. Falling back to non-streaming mode.",
                file=sys.stderr,
            )
            args.stream = False

        result = provider.transform_text_to_json(input_text, schema, stream=args.stream)
        if isinstance(result, Generator):
            display_streaming_response(result)
        else:
            print(json.dumps(result, indent=2))
    except KeyboardInterrupt:
        print("\nProcessing aborted.")
        sys.exit(1)
    except requests.RequestException as e:
        print(f"Error: Failed to communicate with the API. {str(e)}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse API response as JSON. {str(e)}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
