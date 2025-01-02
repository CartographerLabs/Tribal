from setuptools import setup, find_packages

# Read the long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#")
    ]

setup(
    python_requires='>=3.6',
    name="tribal",
    version="0.2.7",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/user1342/tribal",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "tribal-forge=tribal.forge.app:main",
        ],
    },
)
