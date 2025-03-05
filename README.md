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

- [x] Structure output with a schema
- [x] Pipe content directly
- [x] Support for multiple LLM providers: OpenAI, Claude, Mistral, Ollama
- [x] Local LLM support with Ollama
- [x] CLI available as a pip package
- [x] Stream output
- [x] Select different model names from API providers
- [x] `--line` flag to transform each line of input

## Coming Soon

- [ ] Partial JSON output
- [ ] Handling of long content
- [ ] Advanced documentation
- [ ] More providers (e.g., Gemini)
- [ ] `--usage` flag to show usage with token count
- [ ] Python library
- [ ] batch line processing to speed up the process
