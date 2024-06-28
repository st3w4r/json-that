import sys
import json
import os
import requests
from abc import ABC, abstractmethod
from typing import Optional, Dict
import yaml
import argparse
from .version import __version__


class LLMProvider(ABC):
    @abstractmethod
    def transform_text_to_json(
        self, text: str, schema: Optional[str] = None
    ) -> dict[str, any]:
        pass


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def transform_text_to_json(
        self, text: str, schema: Optional[str] = None
    ) -> dict[str, any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        system_message = "transform the raw text to json\n"
        if schema:
            system_message += (
                f" Use the following JSON schema to structure the output: {schema}\n"
            )
            system_message += (
                "Without including extra schema information back in the response."
            )

        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": f"Transform the following text into a JSON format: {text}",
                },
            ],
            "temperature": 0,  # Set temperature to 0 for deterministic output
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=data
        )
        response.raise_for_status()

        result = response.json()
        transformed_text = result["choices"][0]["message"]["content"]

        return json.loads(transformed_text)


class ClaudeProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def transform_text_to_json(
        self, text: str, schema: Optional[str] = None
    ) -> dict[str, any]:
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
            "temperature": 0,  # Set temperature to 0 for deterministic output
        }

        response = requests.post(
            "https://api.anthropic.com/v1/messages", headers=headers, json=data
        )
        response.raise_for_status()

        result = response.json()
        transformed_text = result["content"][0]["text"]

        return json.loads(transformed_text)


class OllamaProvider(LLMProvider):
    def __init__(self, api_url: str = "http://127.0.0.1:11434", model: str = "llama3"):
        self.api_url = api_url
        self.model = model

    def transform_text_to_json(
        self, text: str, schema: Optional[str] = None
    ) -> dict[str, any]:
        headers = {
            "Content-Type": "application/json",
        }

        prompt = f"Transform the following text into a JSON format: {text}\n"
        if schema:
            prompt += f"Use the following JSON schema to structure the output: {schema}\n"
        prompt += "Only output json response without any comment or extra information."

        data = {
            "prompt": prompt,
            "model": self.model,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0,  # Set temperature to 0 for deterministic output
            }
        }

        response = requests.post(
            f"{self.api_url}/api/generate", headers=headers, json=data
        )
        response.raise_for_status()

        result = response.json()
        transformed_text = result["response"]

        return json.loads(transformed_text)


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


def get_provider(config: Config, provider_name: Optional[str] = None) -> LLMProvider:
    if not provider_name:
        provider_name = (
            os.environ.get("LLM_PROVIDER")
            or config.get_default_provider()
            or "openai"
        )

    provider_config = config.get_provider_config(provider_name)

    if provider_name == "ollama":
        api_url = os.environ.get("OLLAMA_API_URL") or provider_config.get("api_url", "http://127.0.0.1:11434")
        model = os.environ.get("OLLAMA_MODEL") or provider_config.get("model", "llama3")
        return OllamaProvider(api_url, model)
    elif provider_name in ["openai", "claude"]:
        api_key = os.environ.get("LLM_API_KEY") or provider_config.get("api_key")
        if not api_key:
            raise ValueError(
                f"API key not found for {provider_name}.\nPlease run the setup command:\njt --setup\n"
            )
        if provider_name == "openai":
            return OpenAIProvider(api_key)
        elif provider_name == "claude":
            return ClaudeProvider(api_key)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider_name}")


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

    while True:
        try:
            provider = input("Choose your LLM provider (openai/claude/ollama): ").lower()
            if provider in ["openai", "claude", "ollama"]:
                break
            print("Invalid choice. Please enter 'openai', 'claude', or 'ollama'.")
        except KeyboardInterrupt:
            print("\nSetup aborted.")
            sys.exit(1)
        except EOFError:
            print("\nSetup aborted.")
            sys.exit(1)

    provider_config = {}

    if provider in ["openai", "claude"]:
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

    set_default = input("Do you want to set this as the default provider? (y/n): ").lower()
    if set_default == 'y':
        config.set_default_provider(provider)

    print(f"Configuration saved to {config.config_file}")


def example():
    print()
    print("examples:")
    print("  echo 'raw text' | jsonthat")
    print("  echo 'raw text' | jt")
    print("  echo 'raw text' | jt --schema schema.json")
    print("  echo 'raw text' | jt --provider openai")
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
    args = parser.parse_args()

    if args.version:
        print(f"jsonthat CLI version {__version__}")
        return

    if args.setup:
        setup_command(config)
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
        result = provider.transform_text_to_json(input_text, schema)
        print(json.dumps(result, indent=2))
    except requests.RequestException as e:
        print(f"Error: Failed to communicate with the API. {str(e)}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse API response as JSON. {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
