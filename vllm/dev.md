# vLLM Local Development Guide

This guide details how to set up, build, test, and lint your local environment for developing vLLM.

---

## 1. Environment Setup

We recommend using [`uv`](https://docs.astral.sh/uv/) for Python package and environment management.

### Install `uv`
If you do not have `uv` installed, run:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Initialize the Virtual Environment
Create a virtual environment with Python 3.12, activate it, and install pre-commit development tooling:
```bash
# Create the environment
uv venv --python 3.12

# Activate the environment
source .venv/bin/activate

# Install linters and development requirements
uv pip install -r requirements/lint.txt

# Install Git pre-commit hooks
pre-commit install
```

> [!WARNING]
> Do not use the system `python3` or bare `pip`/`pip install` commands. All Python commands should execute via `uv` or `.venv/bin/python`.

---

## 2. Installing Dependencies

Depending on whether you are editing only Python files or if you are changing C/C++/CUDA kernels, install dependencies with the appropriate flags:

### Option A: Python-Only Changes (Recommended)
This uses precompiled kernels to significantly speed up your environment setup:
```bash
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

### Option B: C/C++ or CUDA Changes
This builds the C++ / CUDA parts of the codebase from source:
```bash
uv pip install -e . --torch-backend=auto
```

---

## 3. Running Linters & Style Guides

We enforce strict formatting rules, including an **88-character line limit** and **Google-style docstrings**.

### Formatting & Linting Commands
Run formatters/checkers locally before creating a pull request:
```bash
# Run all pre-commit hooks on staged changes
pre-commit run

# Run on all files in the project
pre-commit run --all-files

# Run a specific linter (e.g. ruff)
pre-commit run ruff-check --all-files

# Run mypy type checker (as configured in CI)
pre-commit run mypy-3.12 --all-files --hook-stage manual
```

### Coding Guidelines
* **Docstrings**: Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) (with `Args:`, `Returns:`, and `Raises:` sections). Do not use Sphinx/reStructuredText fields like `:param:` or `:return:`.
* **Style**: Code should be self-documenting. Keep comments concise and tailored to readers who are already familiar with the vLLM architecture.

---

## 4. Running Tests

To run pytest test files locally:

### Install Testing Dependencies
For GPU-based platforms, install the test dependencies:
```bash
# On x86_64:
uv pip install -r requirements/test/cuda.txt

# On other platforms (e.g. ARM/ROCm/etc.):
uv pip install -r requirements/test/cuda.in
```

### Run Tests
```bash
# Run a specific test file
.venv/bin/python -m pytest tests/path/to/test_file.py -v
```

---

## 5. Contribution & Git Commit Best Practices

Before proposing a PR, verify you aren't duplicate-working on an issue or PR that is already in progress:
```bash
gh issue view <issue_number> --repo vllm-project/vllm --comments
gh pr list --repo vllm-project/vllm --state open --search "<issue_number> in:body"
gh pr list --repo vllm-project/vllm --state open --search "<short area keywords>"
```

### Attribution in Commits
Include Co-authored-by trailers in your git commit message where appropriate:
```text
Your descriptive commit message here

Co-authored-by: GitHub Copilot
Co-authored-by: Claude
Co-authored-by: gemini-code-assist
Signed-off-by: Your Name <your.email@example.com>
```

