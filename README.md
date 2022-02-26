# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

# Usage

To run project use:

```
python src/churn_library.py
```

## Using Dependencies

- git-lfs - you should have installed [git-lfs](https://git-lfs.github.com/) to download
  models.
- Python 3.8+
- **RECOMMENDED** pipenv - to easily manage your venv -
  [link](https://pipenv.pypa.io/en/latest/)
- **RECOMMENDED** pyenv - to easily switch between different Python versions -
  [link](https://github.com/pyenv/pyenv)

## Development

Requirements:

- Install [pipenv](https://pipenv.pypa.io/en/latest/) and work in this venv
- Install [pre-commit](https://pre-commit.com/)

After initialization don't forget to use

```
pre-commit install
```

Using Make:

Use `make` to run commands

- `make help` - show help
- `make format` - format code
- `make initialize` - ensure that all files have been downloaded and pre-commit is
  installed
- `make test` - run tests
  - `args="--lf" make test` - run pytest tests with different arguments

For testing purposes you could use function from Makefile:

```
make test_with_cov
```

In coverege folder you can find coverage report.
