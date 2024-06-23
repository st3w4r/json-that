# json that

transform any raw text to json

## installation

```bash
pip install jsonthat
```

## usage

```bash
echo 'my name is jay' | jsonthat
{
  "name": "Jay"
}
```

```bash
options:
    --schema <json_schema_file> pass a json schema to format the output
```
