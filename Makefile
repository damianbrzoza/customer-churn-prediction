help: ## Show this help
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

format: ## Run pre-commit hooks to format code
	@echo "Formatting ..."
	pre-commit run --all-files

args ?= -vvv -cov tests
test: ## Run tests
	pytest $(args)

doc: ## auto doc maker
	rm docs -r || true
	sphinx-quickstart docs --makefile --sep -p project_name -a DamianBrzoza -l en --ext-autodoc --ext-doctest \
	--ext-todo --ext-coverage --ext-viewcode --ext-githubpages --ext-imgmath
	sphinx-apidoc . -o docs/source
	sphinx-build docs/source docs/ -b html
