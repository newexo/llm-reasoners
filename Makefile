# Importable package name (distinct from the "llm-reasoners" distribution name in pyproject.toml)
PACKAGE := reasoners

# Minimum coverage percentage required for tests to pass.
# Low for now: the test suite is a single import smoke test. Raise this as real tests are added.
COVERAGE_FAIL = 1

# Run the test suite
test:
	poetry run pytest

# Format the code using Ruff
format:
	poetry run ruff format .

# Lint the code using Ruff (configured in pyproject.toml [tool.ruff])
lint:
	poetry run ruff check .

# Run all quality checks: formatting, linting, and tests
check: format lint test

# Run tests with coverage enforcement (terminal output only)
# Omit patterns are configured in pyproject.toml [tool.coverage.run].
coverage:
	poetry run coverage run --source=$(PACKAGE) -m pytest
	poetry run coverage report --fail-under=$(COVERAGE_FAIL)

# Run tests with coverage and produce an HTML report
coverage-html:
	poetry run coverage run --source=$(PACKAGE) -m pytest
	poetry run coverage report --fail-under=$(COVERAGE_FAIL)
	poetry run coverage html
	@echo "HTML coverage report generated at htmlcov/index.html"
