#!/usr/bin/env python3
"""
Git Branch Review Expert (C++) - Agente programático
Con capa de abstracción de proveedores LLM.
Implementación inicial: Ollama (local)
"""

import os
import json
import base64
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import requests


# ============================================================
# 1. Abstracción del proveedor de LLM
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
    """Interfaz abstracta que debe implementar cualquier proveedor."""

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        temperature: float = 0.1,
        model: Optional[str] = None,
    ) -> AssistantMessage:
        """
        Realiza una llamada de chat con soporte de tools.
        Debe devolver un AssistantMessage unificado.
        """
        pass


# ============================================================
# 2. Implementación para Ollama (usando API nativa)
# ============================================================

class OllamaProvider(LLMProvider):
    """
    Proveedor Ollama usando la API nativa /api/chat
    (más fiable para tool calling que el endpoint OpenAI-compatible).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = "qwen2.5:14b",  # o llama3.1, qwen3, etc. (modelos con tools)
        timeout: int = 180,
    ):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        temperature: float = 0.1,
        model: Optional[str] = None,
    ) -> AssistantMessage:

        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        if tools:
            payload["tools"] = tools

        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        message = data.get("message", {})
        content = message.get("content") or None

        tool_calls: List[ToolCall] = []
        raw_tool_calls = message.get("tool_calls") or []

        for tc in raw_tool_calls:
            # Ollama formato nativo
            func = tc.get("function", {})
            name = func.get("name")
            arguments = func.get("arguments", {})

            # A veces arguments viene como string
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}

            tool_calls.append(
                ToolCall(
                    id=str(uuid.uuid4()),  # Ollama no siempre devuelve id
                    name=name,
                    arguments=arguments,
                )
            )

        return AssistantMessage(content=content, tool_calls=tool_calls)


# ============================================================
# 3. GitHub API helpers (sin cambios relevantes)
# ============================================================

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_API = "https://api.github.com"
HEADERS = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {GITHUB_TOKEN}" if GITHUB_TOKEN else "",
    "X-GitHub-Api-Version": "2022-11-28",
}


def github_get(url: str, params: dict = None) -> Dict[str, Any]:
    resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
    if resp.status_code == 404:
        raise ValueError(f"Resource not found: {url}")
    resp.raise_for_status()
    return resp.json()


def get_comparison(owner: str, repo: str, base: str, head: str) -> Dict[str, Any]:
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
    url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}"
    data = github_get(url, params={"ref": ref})

    if data.get("encoding") == "base64":
        return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    return data.get("content", "")


def get_raw_diff(owner: str, repo: str, base: str, head: str) -> str:
    url = f"{GITHUB_API}/repos/{owner}/{repo}/compare/{base}...{head}"
    headers = HEADERS.copy()
    headers["Accept"] = "application/vnd.github.v3.diff"
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.text


# ============================================================
# 4. System Prompt (sin cambios)
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
# 5. Tool schemas
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


# ============================================================
# 6. Agente (ahora desacoplado del proveedor)
# ============================================================

def review_branch(
    llmProvider: LLMProvider,
    owner: str,
    repo: str,
    base_branch: str,
    feature_branch: str,
    pr_number: Optional[int] = None,
    model: Optional[str] = None,
) -> str:

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
        assistant_msg = llmProvider.chat(
            messages=messages,
            tools=TOOLS,
            temperature=0.1,
            model=model,
        )

        # Convertimos a formato de mensaje para el historial
        msg_dict: Dict[str, Any] = {"role": "assistant"}

        if assistant_msg.content:
            msg_dict["content"] = assistant_msg.content

        if assistant_msg.has_tool_calls:
            # Formato compatible con la mayoría de proveedores
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in assistant_msg.tool_calls
            ]

        messages.append(msg_dict)

        if not assistant_msg.has_tool_calls:
            return assistant_msg.content or ""

        # Ejecutar tools
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
# 7. Ejemplo de uso
# ============================================================

def main():

    MODEL = os.getenv("MODEL")
    OLLAMA_HOST = os.getenv("OLLAMA_HOST")
    OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY") # TODO: Currently not used, but can be used for future authentication if needed.

    # Create Ollama provider instance
    ollama = OllamaProvider(
        base_url=OLLAMA_HOST,
        default_model=MODEL,
    )

    report = review_branch(
        llmProvider=ollama,
        owner="your-org",
        repo="your-repo",
        base_branch="main",
        feature_branch="feature/new-functionality",
        # pr_number=123,
        # model="llama3.1:70b",   # opcional: sobrescribir modelo
    )

    print(report)

if __name__ == "__main__":
    main()