import pytest
from unittest.mock import patch, mock_open
from jsonthat.main import (
    OpenAIProvider,
    ClaudeProvider,
    OllamaProvider,
    Config,
    get_provider,
    read_schema_file,
)
import os
import json
import yaml


@pytest.fixture
def mock_config():
    return {
        "providers": {
            "openai": {"api_key": "test_openai_key"},
            "claude": {"api_key": "test_claude_key"},
            "ollama": {"api_url": "http://test.ollama.api", "model": "test_model"},
        },
        "default_provider": "openai",
    }


@pytest.fixture
def config(mock_config, tmp_path):
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(mock_config, f)

    with patch(
        "jsonthat.main.Config.get_config_file_path", return_value=str(config_file)
    ):
        return Config()


def test_openai_provider():
    provider = OpenAIProvider("test_key")
    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": '{"key": "value"}'}}]
        }
        result = provider.transform_text_to_json("test text")
        assert result == {"key": "value"}


def test_claude_provider():
    provider = ClaudeProvider("test_key")
    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = {
            "content": [{"text": '{"key": "value"}'}]
        }
        result = provider.transform_text_to_json("test text")
        assert result == {"key": "value"}


def test_ollama_provider():
    provider = OllamaProvider("http://test.api", "test_model")
    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = {"response": '{"key": "value"}'}
        result = provider.transform_text_to_json("test text")
        assert result == {"key": "value"}


def test_config_load(config, mock_config):
    assert config.config == mock_config


def test_config_save(config, tmp_path):
    new_config = {
        "providers": {"newprovider": {"api_key": "new_key"}},
        "default_provider": "newprovider",
    }
    config.config = new_config
    config.save_config()

    with open(config.config_file, "r") as f:
        saved_config = yaml.safe_load(f)

    assert saved_config == new_config


def test_get_provider(config):
    with patch.dict(os.environ, {"LLM_API_KEY": "test_env_key"}):
        provider = get_provider(config, "openai")
        assert isinstance(provider, OpenAIProvider)
        assert provider.api_key == "test_env_key"


def test_get_provider_default(config):
    provider = get_provider(config)
    assert isinstance(provider, OpenAIProvider)
    assert provider.api_key == "test_openai_key"


def test_get_provider_invalid():
    with pytest.raises(ValueError):
        get_provider(Config(), "invalid_provider")


def test_read_schema_file():
    mock_schema = '{"type": "object", "properties": {"name": {"type": "string"}}}'
    with patch("builtins.open", mock_open(read_data=mock_schema)):
        schema = read_schema_file("test_schema.json")
        assert schema == mock_schema


def test_read_schema_file_error():
    with pytest.raises(SystemExit):
        read_schema_file("non_existent_file.json")


def test_openai_provider_json_generation():
    provider = OpenAIProvider("test_key")
    test_input = "My name is Alice and I'm 30 years old"
    expected_output = {"name": "Alice", "age": 30}

    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": json.dumps(expected_output)}}]
        }
        result = provider.transform_text_to_json(test_input)
        assert result == expected_output

        # Verify JSON is mentioned in the prompt
        called_args = mock_post.call_args[1]["json"]
        assert "json" in called_args["messages"][1]["content"].lower()
        assert test_input in called_args["messages"][1]["content"]


def test_claude_provider_json_generation():
    provider = ClaudeProvider("test_key")
    test_input = "The capital of France is Paris"
    expected_output = {"country": "France", "capital": "Paris"}

    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = {
            "content": [{"text": json.dumps(expected_output)}]
        }
        result = provider.transform_text_to_json(test_input)
        assert result == expected_output

        # Verify JSON is mentioned in the prompt
        called_args = mock_post.call_args[1]["json"]
        assert "json" in called_args["messages"][0]["content"].lower()
        assert test_input in called_args["messages"][0]["content"]


def test_ollama_provider_json_generation():
    provider = OllamaProvider("http://test.api", "test_model")
    test_input = "The color of the sky is blue"
    expected_output = {"object": "sky", "property": "color", "value": "blue"}

    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = {
            "response": json.dumps(expected_output)
        }
        result = provider.transform_text_to_json(test_input)
        assert result == expected_output

        # Verify JSON is mentioned in the prompt
        called_args = mock_post.call_args[1]["json"]
        assert "json" in called_args["prompt"].lower()
        assert test_input in called_args["prompt"]


def test_json_generation_with_schema():
    provider = OpenAIProvider("test_key")
    test_input = "John Doe is a software engineer with 5 years of experience"
    test_schema = """{
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "occupation": {"type": "string"},
            "yearsOfExperience": {"type": "number"}
        }
    }"""
    expected_output = {
        "name": "John Doe",
        "occupation": "software engineer",
        "yearsOfExperience": 5,
    }

    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": json.dumps(expected_output)}}]
        }
        result = provider.transform_text_to_json(test_input, test_schema)
        assert result == expected_output

        # Verify JSON and schema are mentioned in the prompt
        called_args = mock_post.call_args[1]["json"]
        assert "json" in called_args["messages"][0]["content"].lower()
        assert "schema" in called_args["messages"][0]["content"].lower()
        assert test_schema in called_args["messages"][0]["content"]


def test_config_display_file_exists(capsys, config):
    config.display_config()
    captured = capsys.readouterr()

    assert f"Config file path: {config.config_file}" in captured.out
    assert "Current configuration:" in captured.out
    assert "providers:" in captured.out
    assert "default_provider:" in captured.out


def test_config_display_file_not_exists(capsys, config):
    with patch("os.path.exists", return_value=False):
        config.display_config()
    captured = capsys.readouterr()

    assert f"Config file path: {config.config_file}" in captured.out
    assert "Config file does not exist" in captured.out
    assert "Run 'jt --setup' to create a configuration" in captured.out


def test_config_display_empty_config(capsys, config):
    config.config = {}
    config.display_config()
    captured = capsys.readouterr()

    assert f"Config file path: {config.config_file}" in captured.out
    assert "Configuration is empty" in captured.out
    assert "Run 'jt --setup' to configure the tool" in captured.out
