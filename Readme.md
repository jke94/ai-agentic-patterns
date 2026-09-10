# AI Agentic Patterns

This repository contains small examples of agentic patterns built around Ollama Cloud / remote hosted LLM endpoints and GitHub-aware workflows.

## Included examples

### 1) Orchestrator-Workers pattern
File: `src/orchestrator-worker-pattern.py`

This example demonstrates a classic orchestration pattern:

- an orchestrator receives the task,
- distributes it to multiple workers,
- each worker reasons with a different prompt,
- then the orchestrator synthesizes the results into a final answer.

This is useful when you want several specialized perspectives on the same input before generating a final response.

### 2) Programmatic PR Reviewer agent
File: `src/programmatic-agent-PR-reviewer.py`

This script acts like an automated code reviewer for a GitHub branch comparison.

It:

- compares a feature branch against a base branch,
- retrieves changed files and patches through the GitHub API,
- optionally reads file contents from a specific ref,
- asks an LLM to review technical risk using a strict review rubric.

The goal is to produce a concise, evidence-based review focused on memory safety, concurrency, API compatibility, architecture, and regression risk.

---

## Technology stack

- Python 3
- Ollama Cloud / hosted Ollama endpoint for LLM access
- GitHub REST API for branch and PR diff data
- python-dotenv for environment configuration

---

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
. .venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with the necessary values.

Example:

```env
MODEL=qwen2.5:7b
OLLAMA_HOST=https://your-ollama-cloud-endpoint.example
OLLAMA_API_KEY=your_cloud_api_key
GITHUB_TOKEN=your_github_token
GITHUB_OWNER=your_org_or_user
GITHUB_REPOSITORY=your_repo
BASE_BRANCH=main
FEATURE_BRANCH=feature/my-branch
```

> `OLLAMA_HOST` should point to the Ollama Cloud / remote endpoint you are using, not a local localhost address. `GITHUB_TOKEN` is required only when GitHub API requests need authentication or when the repository is private.

---

## Run the examples

### Orchestrator-Workers example

```bash
python src/orchestrator-worker-pattern.py
```

### PR review agent

```bash
python src/programmatic-agent-PR-reviewer.py
```

---

## Notes

- The repository is intentionally focused on patterns and experimentation.
- The PR reviewer is a programmatic agent prototype and is designed to be adapted for more complex review workflows.
- The examples assume an Ollama Cloud / remote HTTP endpoint and are designed to work with hosted deployment environments.

---

## Project structure

```text
ai-agentic-patterns/
├── .env
├── Readme.md
├── requirements.txt
├── src/
│   ├── orchestrator-worker-pattern.py
│   └── programmatic-agent-PR-reviewer.py
└── .venv/
```