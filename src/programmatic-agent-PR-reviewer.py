#!/usr/bin/env python3
"""
Git Branch Review Expert (C++) - Agente programático
Usa únicamente la API REST de GitHub (sin clonar, sin PyGithub)
"""

import os
import json
import base64
from typing import Optional, Dict, Any, List
import requests
from openai import OpenAI   # o anthropic / xai según prefieras

# ====================== Configuración ======================
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_API = "https://api.github.com"
HEADERS = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {GITHUB_TOKEN}" if GITHUB_TOKEN else "",
    "X-GitHub-Api-Version": "2022-11-28"
}

client = OpenAI()   # Cambia por el SDK que uses


# ====================== Helpers de la API de GitHub ======================
def github_get(url: str, params: dict = None) -> Dict[str, Any]:
    resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
    if resp.status_code == 404:
        raise ValueError(f"Recurso no encontrado: {url}")
    resp.raise_for_status()
    return resp.json()


def get_comparison(owner: str, repo: str, base: str, head: str) -> Dict[str, Any]:
    """
    Equivalente a: git diff base...head + git log base..head
    Endpoint: GET /repos/{owner}/{repo}/compare/{base}...{head}
    """
    url = f"{GITHUB_API}/repos/{owner}/{repo}/compare/{base}...{head}"
    data = github_get(url)

    files = []
    for f in data.get("files", []):
        files.append({
            "filename": f["filename"],
            "status": f["status"],                    # added, modified, removed, renamed
            "additions": f.get("additions", 0),
            "deletions": f.get("deletions", 0),
            "changes": f.get("changes", 0),
            "previous_filename": f.get("previous_filename"),
            "patch": f.get("patch")                   # unified diff (puede ser None)
        })

    commits = [
        {
            "sha": c["sha"][:8],
            "message": c["commit"]["message"].split("\n")[0],
            "author": c["commit"]["author"]["name"]
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
        "html_url": data.get("html_url")
    }


def get_pr_files(owner: str, repo: str, pr_number: int) -> Dict[str, Any]:
    """
    Cuando existe un Pull Request → más preciso y completo.
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
            "patch": f.get("patch")
        })

    # También obtenemos info del PR
    pr_url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_number}"
    pr = github_get(pr_url)

    return {
        "title": pr.get("title"),
        "body": pr.get("body"),
        "state": pr.get("state"),
        "base": pr["base"]["ref"],
        "head": pr["head"]["ref"],
        "html_url": pr.get("html_url"),
        "files": files
    }


def get_file_content(owner: str, repo: str, path: str, ref: str) -> str:
    """
    Lee el contenido completo de un archivo en una rama concreta.
    Endpoint: GET /repos/{owner}/{repo}/contents/{path}?ref={ref}
    """
    url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}"
    data = github_get(url, params={"ref": ref})

    if data.get("encoding") == "base64":
        return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    return data.get("content", "")


def get_raw_diff(owner: str, repo: str, base: str, head: str) -> str:
    """
    Diff completo en formato texto (útil cuando los patches individuales están truncados).
    """
    url = f"{GITHUB_API}/repos/{owner}/{repo}/compare/{base}...{head}"
    headers = HEADERS.copy()
    headers["Accept"] = "application/vnd.github.v3.diff"
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.text


# ====================== System Prompt (el original del agente) ======================
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


# ====================== Tools schema para el LLM ======================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_comparison",
            "description": "Obtiene el resumen completo de cambios entre dos ramas (archivos, patches, commits). Úsalo siempre primero.",
            "parameters": {
                "type": "object",
                "properties": {
                    "base": {"type": "string", "description": "Rama base (ej: main)"},
                    "head": {"type": "string", "description": "Rama a revisar (feature)"}
                },
                "required": ["base", "head"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_pr_files",
            "description": "Si existe un Pull Request, usa este tool (más preciso). Devuelve los archivos cambiados + patches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pr_number": {"type": "integer"}
                },
                "required": ["pr_number"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_file_content",
            "description": "Lee el contenido completo de un archivo en una rama concreta (útil para ver contexto, headers, etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "ref": {"type": "string", "description": "Rama o SHA (normalmente la feature branch)"}
                },
                "required": ["path", "ref"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_raw_diff",
            "description": "Obtiene el diff completo en formato texto plano (usar cuando los patches individuales estén truncados).",
            "parameters": {
                "type": "object",
                "properties": {
                    "base": {"type": "string"},
                    "head": {"type": "string"}
                },
                "required": ["base", "head"]
            }
        }
    }
]


# ====================== Bucle del agente ======================
def review_branch(
    owner: str,
    repo: str,
    base_branch: str,
    feature_branch: str,
    pr_number: Optional[int] = None,
    model: str = "gpt-4o"
) -> str:

    # Inyectamos contexto fijo
    context = {
        "owner": owner,
        "repo": repo,
        "base_branch": base_branch,
        "feature_branch": feature_branch,
        "pr_number": pr_number
    }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Revisa la rama `{feature_branch}` contra `{base_branch}` "
                f"del repositorio `{owner}/{repo}`.\n"
                f"{'Existe el Pull Request #' + str(pr_number) if pr_number else 'No se ha indicado número de PR.'}\n\n"
                "Sigue estrictamente el algoritmo y el formato de salida definidos."
            )
        }
    ]

    while True:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.1
        )

        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            return msg.content

        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            try:
                if name == "get_comparison":
                    result = get_comparison(owner, repo, args["base"], args["head"])
                elif name == "get_pr_files":
                    result = get_pr_files(owner, repo, args["pr_number"])
                elif name == "get_file_content":
                    result = get_file_content(owner, repo, args["path"], args["ref"])
                elif name == "get_raw_diff":
                    result = get_raw_diff(owner, repo, args["base"], args["head"])
                else:
                    result = f"Tool desconocida: {name}"
            except Exception as e:
                result = f"Error en la tool {name}: {str(e)}"

            # Limitamos tamaño de respuesta de tools para no saturar el contexto
            content = json.dumps(result, indent=2, ensure_ascii=False) if isinstance(result, (dict, list)) else str(result)
            if len(content) > 25000:
                content = content[:25000] + "\n\n... [truncado por tamaño]"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": content
            })


# ====================== Ejemplo de uso ======================
if __name__ == "__main__":
    report = review_branch(
        owner="tu-org",
        repo="tu-repo",
        base_branch="main",
        feature_branch="feature/nueva-funcionalidad",
        # pr_number=123,          # descomenta si tienes número de PR
        model="gpt-4o"
    )
    print(report)