from setuptools import setup, find_packages

# Read the requirements from the requirements.txt file
with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name="Tribal",
    version="0.13",
    packages=find_packages(),
    # Optional metadata
    author="James Stevenson",
    author_email="opensource@jamesstevenson.me",
    description="ML for countering violent extremism",
    url="https://github.com/CartographerLabs/Tribal",
    install_requires=requirements,  # Dependencies from requirements.txt
)
