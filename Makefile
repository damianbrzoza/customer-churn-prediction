help: ## Show this help
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

initialize: ## Start pre-commit and ensure that all files from git-lfs are downloaded
	pre-commit install
	git-lfs pull

format: ## Run pre-commit hooks to format code
	@echo "Formatting ..."
	pre-commit run --all-files

args ?= -vvv -cov tests
test: ## Run tests
	pytest $(args)

test_with_cov: ## Run tests with coverage report in html
	pytest tests --cov=src --cov-report html:coverage/
