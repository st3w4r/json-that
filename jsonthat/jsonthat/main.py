import sys
import os
from openai import OpenAI
import json
import argparse


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("OpenAI Api key env var is not set\n")
    print("Please set it using:")
    print("export OPENAI_API_KEY=your-api-key")
    sys.exit(1)

assert OPENAI_API_KEY, "OPENAI_API_KEY is not set"

client = OpenAI(api_key=OPENAI_API_KEY)

INDENT_SIZE = 2


def transform_to_json(text: str, schema: str = None):
    if schema:
        messages = [
            {
                "role": "system",
                "content": "transform the raw text to json",
            },
            {
                "role": "system",
                "content": "usnig this schema:\n" + schema,
            },
            {
                "role": "system",
                "content": "without including extra schema inforamtion back in the response",
            },
            {
                "role": "user",
                "content": text,
            },
        ]
    else:
        messages = [
            {
                "role": "system",
                "content": "transform the raw text to json",
            },
            {
                "role": "user",
                "content": text,
            },
        ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    json_data = response.choices[0].message.content
    return json_data


def read_stdin(schema: str = None):
    try:
        input_data = sys.stdin.read()
        res = transform_to_json(input_data, schema)

        try:
            res = json.loads(res)
            print(json.dumps(res, indent=INDENT_SIZE))
        except json.JSONDecodeError:
            print("Malformed genertated JSON")
            sys.exit(1)
    except KeyboardInterrupt:
        print("Exiting...")
        sys.exit(0)


def read_schemafile(schema_filepath: str):
    with open(schema_filepath, "r") as f:
        schema = f.read()
    return schema


def example():
    print()
    print("Examples:")
    print("  echo 'raw text' | jsonthat")
    print("  echo 'raw text' | jt")
    print("  echo 'raw text' | jsonthat -s schema.json")

    print("""
  echo 'my name is jay' | jsonthat
  {
    "name": "Jay"
  }
          """)
    print()


class CustomHelpParser(argparse.ArgumentParser):
    def print_help(self):
        super().print_help()
        example()


def main():
    parser = CustomHelpParser()
    parser.add_argument("-s", "--schema", help="input json schema file")

    args = sys.argv[1:]
    try:
        args = parser.parse_args(args)
    except argparse.ArgumentError:
        parser.print_help()
        return

    if sys.stdin.isatty():
        parser.print_help()
        return

    schema_filepath = args.schema
    if schema_filepath:
        schema = read_schemafile(schema_filepath)
    else:
        schema = None

    read_stdin(schema)


if __name__ == "__main__":
    main()
