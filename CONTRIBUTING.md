# Contributing to SteelFlow

Thanks for your interest in SteelFlow. We welcome issues and pull requests.

## How to Contribute

- Open an issue to discuss a feature or bug before large changes.
- Keep PRs focused and small when possible.
- Add or update tests when you change behavior.
- Follow existing code style and naming conventions.

## Development Setup

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

## Running Tests

```bash
pytest
```

## Reporting Bugs

Please include:

- OS, CPU/GPU details
- Model/backend used (e.g., vLLM, llama.cpp)
- Exact error logs or stack traces
- Steps to reproduce

## Code of Conduct

Be respectful and constructive. Harassment or abusive behavior will not be tolerated.
