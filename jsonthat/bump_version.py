import toml


def read_pyproject_data():
    with open("pyproject.toml", "r") as f:
        pyproject_data = toml.load(f)
    return pyproject_data


def get_field(pyproject_data, field_path):
    fields = field_path.split(".")
    value = pyproject_data
    for field in fields:
        value = value.get(field, None)
        if value is None:
            raise ValueError(f"{field_path} not found in pyproject.toml")
    return value


def write_version_to_file(name, version):
    version_file_content = f'__version__ = "{version}"\n'
    with open(f"{name}/version.py", "w") as f:
        f.write(version_file_content)


def main():
    pyproject_data = read_pyproject_data()
    name = get_field(pyproject_data, "tool.poetry.name")
    version = get_field(pyproject_data, "tool.poetry.version")
    write_version_to_file(name, version)
    print(f"Version {version} written to {name}/version.py")


if __name__ == "__main__":
    main()
