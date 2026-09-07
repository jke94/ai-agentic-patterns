#!/usr/bin/env python3
"""
Git Branch Review Expert (C++) - Programmatic Agent

LLM provider abstraction layer.
Initial implementation: Ollama (local or remote/web).
"""

import os
import json
import base64
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import requests
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 0. GitHub API helpers
# ============================================================

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_API = "https://api.github.com"
HEADERS = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {GITHUB_TOKEN}" if GITHUB_TOKEN else "",
    "X-GitHub-Api-Version": "2022-11-28",
}

# ============================================================
# 1. System Prompt
# ============================================================

SYSTEM_PROMPT = """
# VS Code Agent: Git Branch Review Expert (C++)

## Objective
Compare a proposed branch against the repository's default branch (main/master) before integration and produce a concise, evidence-based review focused on technical risk.

## Role
Act as:
- Senior Software Architect
- Modern C++ Expert (C++17 where applicable)
- Code Quality and Maintainability Reviewer
- Git and Pull Request Review Specialist

## Git Comparison Algorithm (Mandatory)
You have tools that give you the equivalent of:
- git diff --name-status base...feature
- git log base..feature --oneline
- full patches of changed files
- ability to read any file content on either branch

Always start by calling the comparison tool.

## Review Criteria (Priority Order)
1. Ownership, Lifetime & Memory (Highest priority)
2. Concurrency & Thread Safety
3. Architecture
4. Public ABI / API Surface
5. Tests & Regression Analysis
6. C++ Quality
7. Engineering Best Practices
8. Risk Assessment

## Evidence Requirements
Every finding must include:
- Affected file
- Code snippet or change reference
- Technical explanation
- Impact
- Severity (CRITICAL / HIGH / MEDIUM / LOW)

## Output Format (strict)
### Executive Summary
- Number of critical findings
- Number of high findings
- Number of medium findings
- Integration Risk Score: <N>/100 (<Level>)
- Final recommendation: APPROVE | APPROVE WITH COMMENTS | CHANGES REQUIRED

### Findings
#### [SEVERITY] Short Title
**Evidence**
- File:
- Change:

**Issue**
...

**Impact**
...

**Recommendation**
...

### ABI / API Impact
### Tests & Regression Risk
### Architectural Risks
### Positive Aspects
### Integration Risk Score Detail
- Score: N/100
- Level: ...
- Justification: ...

Be concise. Prioritize ownership, concurrency, architecture, ABI/API and tests.
"""

# ============================================================
# 2. Tool schemas
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_comparison",
            "description": "Get the complete summary of changes between two branches (files, patches, commits). Always use this first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "base": {"type": "string", "description": "Base branch (e.g. main)"},
                    "head": {"type": "string", "description": "Branch to review (feature)"},
                },
                "required": ["base", "head"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_pr_files",
            "description": "If a Pull Request exists, use this tool (more precise). Returns changed files + patches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pr_number": {"type": "integer"},
                },
                "required": ["pr_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_file_content",
            "description": "Read the complete content of a file on a specific branch.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "ref": {"type": "string", "description": "Branch or SHA"},
                },
                "required": ["path", "ref"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_raw_diff",
            "description": "Get the complete diff in plain text (use when individual patches are truncated).",
            "parameters": {
                "type": "object",
                "properties": {
                    "base": {"type": "string"},
                    "head": {"type": "string"},
                },
                "required": ["base", "head"],
            },
        },
    },
]

def github_get(url: str, params: dict = None) -> Dict[str, Any]:
    resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
    if resp.status_code == 404:
        raise ValueError(f"Resource not found: {url}")
    resp.raise_for_status()
    return resp.json()


def get_comparison(owner: str, repo: str, base: str, head: str) -> Dict[str, Any]:
    """
    Equivalent to: git diff base...head + git log base..head
    Endpoint: GET /repos/{owner}/{repo}/compare/{base}...{head}
    """
    url = f"{GITHUB_API}/repos/{owner}/{repo}/compare/{base}...{head}"
    data = github_get(url)

    files = []
    for f in data.get("files", []):
        files.append({
            "filename": f["filename"],
            "status": f["status"],
            "additions": f.get("additions", 0),
            "deletions": f.get("deletions", 0),
            "changes": f.get("changes", 0),
            "previous_filename": f.get("previous_filename"),
            "patch": f.get("patch"),
        })

    commits = [
        {
            "sha": c["sha"][:8],
            "message": c["commit"]["message"].split("\n")[0],
            "author": c["commit"]["author"]["name"],
        }
        for c in data.get("commits", [])
    ]

    return {
        "status": data.get("status"),
        "ahead_by": data.get("ahead_by"),
        "behind_by": data.get("behind_by"),
        "total_commits": data.get("total_commits"),
        "commits": commits,
        "files": files,
        "html_url": data.get("html_url"),
    }


def get_pr_files(owner: str, repo: str, pr_number: int) -> Dict[str, Any]:
    """
    When a Pull Request exists, this is more precise and complete.
    Endpoint: GET /repos/{owner}/{repo}/pulls/{pr_number}/files
    """
    url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_number}/files"
    files_data = github_get(url)

    files = []
    for f in files_data:
        files.append({
            "filename": f["filename"],
            "status": f["status"],
            "additions": f.get("additions", 0),
            "deletions": f.get("deletions", 0),
            "changes": f.get("changes", 0),
            "previous_filename": f.get("previous_filename"),
            "patch": f.get("patch"),
        })

    pr_url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_number}"
    pr = github_get(pr_url)

    return {
        "title": pr.get("title"),
        "body": pr.get("body"),
        "state": pr.get("state"),
        "base": pr["base"]["ref"],
        "head": pr["head"]["ref"],
        "html_url": pr.get("html_url"),
        "files": files,
    }


def get_file_content(owner: str, repo: str, path: str, ref: str) -> str:
    """
    Read the complete content of a file on a specific branch.
    Endpoint: GET /repos/{owner}/{repo}/contents/{path}?ref={ref}
    """
    url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}"
    data = github_get(url, params={"ref": ref})

    if data.get("encoding") == "base64":
        return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    return data.get("content", "")


def get_raw_diff(owner: str, repo: str, base: str, head: str) -> str:
    """
    Complete diff in plain text format (useful when individual patches are truncated).
    """
    url = f"{GITHUB_API}/repos/{owner}/{repo}/compare/{base}...{head}"
    headers = HEADERS.copy()
    headers["Accept"] = "application/vnd.github.v3.diff"
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.text


# ============================================================
# 3. LLM Provider Abstraction
# ============================================================

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class AssistantMessage:
    content: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class LLMProvider(ABC):
    """Abstract interface that any LLM provider must implement."""

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        temperature: float = 0.1,
    ) -> AssistantMessage:
        """
        Perform a chat completion with tool support.
        Must return a unified AssistantMessage.
        """
        pass


# ============================================================
# 4. Ollama Provider (local or remote/web)
# ============================================================

class OllamaProvider(LLMProvider):
    """
    Ollama provider using the native /api/chat endpoint.
    Supports both local instances and remote/web Ollama deployments.
    API key is optional (required for some hosted/remote instances).
    """

    def __init__(
        self,
        base_url: str,
        default_model: str,
        api_key: Optional[str] = None,
        timeout: int = 180,
    ):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.api_key = api_key
        self.timeout = timeout

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        temperature: float = 0.1,
    ) -> AssistantMessage:

        payload = {
            "model": self.default_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        if tools:
            payload["tools"] = tools

        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        if not resp.ok:
            raise RuntimeError(
                f"Ollama API error ({resp.status_code}) at {self.base_url}/api/chat: "
                f"{resp.text}"
            )
        data = resp.json()

        message = data.get("message", {})
        content = message.get("content") or None

        tool_calls: List[ToolCall] = []
        raw_tool_calls = message.get("tool_calls") or []

        for tc in raw_tool_calls:
            func = tc.get("function", {})
            name = func.get("name")
            arguments = func.get("arguments", {})

            # Arguments may arrive as a JSON string
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}

            tool_calls.append(
                ToolCall(
                    id=str(uuid.uuid4()),  # Ollama does not always return an id
                    name=name,
                    arguments=arguments,
                )
            )

        return AssistantMessage(content=content, tool_calls=tool_calls)

# ============================================================
# 5. Agent (decoupled from the concrete LLM provider)
# ============================================================

def review_branch(
    llm_provider: LLMProvider,
    owner: str,
    repo: str,
    base_branch: str,
    feature_branch: str,
    pr_number: Optional[int] = None,
) -> str:
    """
    Run the branch review agent.
    The model is configured inside the injected LLMProvider instance.
    """

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Review the `{feature_branch}` branch against `{base_branch}` "
                f"in the `{owner}/{repo}` repository.\n"
                f"{'Pull Request #' + str(pr_number) + ' exists.' if pr_number else 'No PR number was provided.'}\n\n"
                "Strictly follow the defined algorithm and output format."
            ),
        },
    ]

    while True:
        assistant_msg = llm_provider.chat(
            messages=messages,
            tools=TOOLS,
            temperature=0.1,
        )

        # Convert to a message dict for the conversation history
        msg_dict: Dict[str, Any] = {"role": "assistant"}

        if assistant_msg.content:
            msg_dict["content"] = assistant_msg.content

        if assistant_msg.has_tool_calls:
            # Format compatible with most providers
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    },
                }
                for tc in assistant_msg.tool_calls
            ]

        messages.append(msg_dict)

        if not assistant_msg.has_tool_calls:
            return assistant_msg.content or ""

        # Execute tools
        for tc in assistant_msg.tool_calls:
            try:
                if tc.name == "get_comparison":
                    result = get_comparison(owner, repo, tc.arguments["base"], tc.arguments["head"])
                elif tc.name == "get_pr_files":
                    result = get_pr_files(owner, repo, tc.arguments["pr_number"])
                elif tc.name == "get_file_content":
                    result = get_file_content(owner, repo, tc.arguments["path"], tc.arguments["ref"])
                elif tc.name == "get_raw_diff":
                    result = get_raw_diff(owner, repo, tc.arguments["base"], tc.arguments["head"])
                else:
                    result = f"Unknown tool: {tc.name}"
            except Exception as e:
                result = f"Error in tool {tc.name}: {str(e)}"

            content = (
                json.dumps(result, indent=2, ensure_ascii=False)
                if isinstance(result, (dict, list))
                else str(result)
            )
            if len(content) > 25000:
                content = content[:25000] + "\n\n... [truncated due to size]"

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": content,
            })


# ============================================================
# 6. Example usage
# ============================================================

def main():

    model = os.getenv("MODEL")
    ollama_host = os.getenv("OLLAMA_HOST")
    ollama_api_key = os.getenv("OLLAMA_API_KEY")

    github_owner = os.getenv("GITHUB_OWNER")
    github_repo = os.getenv("GITHUB_REPOSITORY")
    base_branch = os.getenv("BASE_BRANCH")
    feature_branch = os.getenv("FEATURE_BRANCH")

    required_settings = {
        "MODEL": model,
        "OLLAMA_HOST": ollama_host,
        "GITHUB_OWNER": github_owner,
        "GITHUB_REPOSITORY": github_repo,
        "BASE_BRANCH": base_branch,
        "FEATURE_BRANCH": feature_branch,
    }
    missing_settings = [
        name for name, value in required_settings.items() if not value
    ]
    if missing_settings:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing_settings)
        )

    # Create Ollama provider instance (model is injected here)
    ollama = OllamaProvider(
        base_url=ollama_host,
        default_model=model,
        api_key=ollama_api_key,
        timeout=180,
    )

    report = review_branch(
        llm_provider=ollama,
        owner=github_owner,
        repo=github_repo,
        base_branch=base_branch,
        feature_branch=feature_branch,
        # pr_number=123,
    )

    print(report)

if __name__ == "__main__":
    main()
