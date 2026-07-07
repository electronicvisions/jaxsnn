# Building dist packages and run tests

First create a virtual environment.
```shell
set -o errexit
python3 -m venv venv
source venv/bin/activate
```

After that install jaxsnn via pip.
```shell
python3 -m pip install --upgrade pip
python3 -m pip install .
```

You might also want to execute the tests for the simulator-only mode
(no BrainScaleS-2 tests)
```shell
python3 -m pip install .[test-nir]
pytest
```

Or even execute tests with the development version of NIR:
```shell
python3 -m pip install --upgrade -r test-dev-requirements.txt
pytest
```
