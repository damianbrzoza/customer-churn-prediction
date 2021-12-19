## Development

Requirements:

- Install `pipenv` (https://pipenv.pypa.io/en/latest/) and work in this venv
- Rename project_name in Makefile and setup.py

Using Make:

Use `make` to run commands

- `make help` - show help
- `make format` - format code
- `make test` - run tests
  - `args="--lf" make test` - run pytest tests with different arguments
- `make docs` - automatically make documentation
