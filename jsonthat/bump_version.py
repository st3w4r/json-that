import toml
import os

def read_version_from_pyproject():
    with open("pyproject.toml", "r") as f:
        pyproject_data = toml.load(f)
    version = pyproject_data.get("tool", {}).get("poetry", {}).get("version", None)
    if version is None:
        raise ValueError("Version not found in pyproject.toml")
    return version

def write_version_to_file(version):
    version_file_content = f'__version__ = "{version}"\n'
    with open("jsonthat/version.py", "w") as f:
        f.write(version_file_content)

def main():
    version = read_version_from_pyproject()
    write_version_to_file(version)
    print(f"Version {version} written to version.py")

if __name__ == "__main__":
    main()
