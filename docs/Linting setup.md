## How to setup linting

Install the extension
https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff for linting and formatting of python code
 
Install ```pre-commit```  with ```pip install pre-commit```, or your choice of installing python packages.
In the root of the repo, run ```pre-commit install``` to install the spellchecker and linter check for every commit you make. 
Note: the first commit can take ~30 seconds as it installs the tools

When a commit fails due to spelling or formatting mistakes, press ```Show command output``` to see the actual error. There is nothing wrong with your git! Fix the errors and commit again