# json that

transform any raw text to json, using LLM.

Provider:
- OpenAI
- Claude AI
- Mistral AI
- Ollama (local)

## installation

```bash
pip install jsonthat
```

```bash
jt --setup
```

## usage

```bash
echo 'my name is jay' | jt
{
  "name": "Jay"
}
```

```bash
options:
    --schema   <json_schema_file> Pass a json schema to format the output
    --provider <provider>         Specify the LLM provider to use
    --stream                      Stream the output
```
