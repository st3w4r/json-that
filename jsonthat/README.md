# json that

transform any raw text to json

## installation

```bash
pip install jsonthat
```

```bash
export OPENAI_API_KEY=<your_openai_api_key>
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
