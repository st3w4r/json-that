# json that


```bash
pip install jsonthat
```

```bash
jt --setup
```

```bash
echo 'my name is jay' | jt
{
  "name": "Jay"
}
```


## Features

- [x] Schema to structure output
- [x] Pipe content directly
- [x] Multi-LLM provider support: OpenAI, Claude, Mistral, Ollama
- [x] Local LLM with Ollama
- [x] CLI released as pip package
- [x] Stream the output

## Coming

- [ ] partial json output
- [ ] handle long content
- [ ] advance documentation
- [ ] more providers
  - [ ] gemini
- [ ] select different model name from api provider
- [ ] flag `--line` to transform each line of the input
- [ ] flag `--usage` to show usage with tokens count
- [ ] library for python
