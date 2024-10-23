import subprocess
from setuptools import setup, find_packages

# Read the requirements from the requirements.txt file
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

# Separate the Git-based requirements
git_requirements = [req for req in requirements if req.startswith("git+")]
other_requirements = [req for req in requirements if not req.startswith("git+")]

# Install Git dependencies manually
if git_requirements:
    for req in git_requirements:
        print(f"Installing {req}")
        subprocess.check_call(["pip", "install", req])

setup(
    name="Tribal",
    version="0.32",
    packages=find_packages(),
    # Optional metadata
    author="James Stevenson",
    author_email="opensource@jamesstevenson.me",
    description="ML for countering violent extremism",
    url="https://github.com/CartographerLabs/Tribal",
    install_requires=other_requirements,  # Dependencies without GitHub URLs
)
