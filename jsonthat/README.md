# json that

transform any raw text to json, using LLM.

Provider:
- OpenAI
- Claude AI

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
    --schema <json_schema_file> pass a json schema to format the output
```
