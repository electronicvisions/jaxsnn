# Building dist packages and run tests

```shell
set -o errexit
python -m virtualenv venv
source venv/bin/activate
pip install --upgrade pip
pip install --upgrade build
pip install --upgrade virtualenv
sed -i 's/name = "jaxsnn"/name = "jaxsnn"\nversion = "0.0.999"/' pyproject.toml
python3 -m build
pip install dist/*.whl
shopt -s globstar
for test in tests/**/*.py; do python $test; done
```
