import sys
import json
import os
import requests
from abc import ABC, abstractmethod
from typing import Optional
import yaml
import argparse


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
        return {}

    def save_config(self):
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        with open(self.config_file, "w") as f:
            yaml.dump(self.config, f)

    def get(self, key, fallback=None):
        return self.config.get(key, fallback)

    def set(self, key, value):
        self.config[key] = value
        self.save_config()


def get_provider(config: Config) -> LLMProvider:
    provider_name = (
        os.environ.get("LLM_PROVIDER") or config.get("provider", "openai").lower()
    )
    api_key = os.environ.get("LLM_API_KEY") or config.get("api_key")

    if not api_key:
        raise ValueError(
            "API key not found.\nPlease run the setup command:\njt --setup\n\nor\n\nSet the LLM_API_KEY and LLM_PROVIDER environment variable."
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


def setup_command():
    config = Config()

    print("Welcome to 'json that' CLI setup!")

    while True:
        try:
            provider = input("Choose your LLM provider (openai/claude): ").lower()
            if provider in ["openai", "claude"]:
                break
            print("Invalid choice. Please enter 'openai' or 'claude'.")
        except KeyboardInterrupt:
            print("\nSetup aborted.")
            sys.exit(1)
        except EOFError:
            print("\nSetup aborted.")
            sys.exit(1)

    try:
        api_key = input(f"Enter your {provider.capitalize()} API key: ")
    except KeyboardInterrupt:
        print("\nSetup aborted.")
        sys.exit(1)
    except EOFError:
        print("\nSetup aborted.")
        sys.exit(1)

    config.set("provider", provider)
    config.set("api_key", api_key)

    print(f"Configuration saved to {config.config_file}")


def example():
    print()
    print("examples:")
    print("  echo 'raw text' | jsonthat")
    print("  echo 'raw text' | jt")
    print("  echo 'raw text' | jt -s schema.json")
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
    parser = CustomHelpParser(description="Text to JSON CLI")
    parser.add_argument("--setup", action="store_true", help="Run the setup command")
    parser.add_argument("--schema", type=str, help="Path to the schema file")
    args = parser.parse_args()

    if args.setup:
        setup_command()
        return

    if sys.stdin.isatty():
        parser.print_help()
        return

    config = Config()

    try:
        provider = get_provider(config)
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
