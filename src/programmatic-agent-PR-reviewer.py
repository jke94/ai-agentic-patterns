"""
Git branch review agent for C++ codebases.

This script compares a feature branch against the base branch (main/master) and generates a structured review report focusing on technical risks, ownership, concurrency, architecture, ABI/API, and tests.
"""

import os
import json
import base64
from typing import Optional, Dict, Any, List
import requests
from openai import OpenAI   # or anthropic / xai, as preferred

# ====================== Configuration ======================
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_API = "https://api.github.com"
HEADERS = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {GITHUB_TOKEN}" if GITHUB_TOKEN else "",
    "X-GitHub-Api-Version": "2022-11-28"
}

client = OpenAI()   # Replace with the SDK you use


# ====================== GitHub API helpers ======================
def github_get(url: str, params: dict = None) -> Dict[str, Any]:
    resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
    if resp.status_code == 404:
        raise ValueError(f"Resource not found: {url}")
    resp.raise_for_status()
    return resp.json()


def get_comparison(
        owner: str,
        repo: str,
        base: str,
        head: str
    ) -> Dict[str, Any]:
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
            "status": f["status"],                    # added, modified, removed, renamed
            "additions": f.get("additions", 0),
            "deletions": f.get("deletions", 0),
            "changes": f.get("changes", 0),
            "previous_filename": f.get("previous_filename"),
            "patch": f.get("patch")                   # unified diff (may be None)
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
            "patch": f.get("patch")
        })

    # Also retrieve PR information
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
    Complete diff in text format (useful when individual patches are truncated).
    """
    url = f"{GITHUB_API}/repos/{owner}/{repo}/compare/{base}...{head}"
    headers = HEADERS.copy()
    headers["Accept"] = "application/vnd.github.v3.diff"
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.text


# ====================== System Prompt (the agent's original prompt) ======================
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


# ====================== Tool schema for the LLM ======================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_comparison",
            "description": "Get the complete summary of changes between two branches (files, patches, commits). Always use this first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "base": {"type": "string", "description": "Base branch (for example, main)"},
                    "head": {"type": "string", "description": "Branch to review (feature)"}
                },
                "required": ["base", "head"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_pr_files",
            "description": "If a Pull Request exists, use this tool (more precise). Returns changed files and patches.",
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
            "description": "Read the complete content of a file on a specific branch (useful for viewing context, headers, etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "ref": {"type": "string", "description": "Branch or SHA (usually the feature branch)"}
                },
                "required": ["path", "ref"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_raw_diff",
            "description": "Get the complete diff in plain text format (use when individual patches are truncated).",
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


# ====================== Agent loop ======================
def review_branch(
    owner: str,
    repo: str,
    base_branch: str,
    feature_branch: str,
    pr_number: Optional[int] = None,
    model: str = "gpt-4o"
) -> str:

    # Inject fixed context
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
                f"Review the `{feature_branch}` branch against `{base_branch}` "
                f"in the `{owner}/{repo}` repository.\n"
                f"{'Pull Request #' + str(pr_number) + ' exists.' if pr_number else 'No PR number was provided.'}\n\n"
                "Strictly follow the defined algorithm and output format."
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
                    result = f"Unknown tool: {name}"
            except Exception as e:
                result = f"Error in tool {name}: {str(e)}"

            # Limit tool response size to avoid overloading the context
            content = json.dumps(result, indent=2, ensure_ascii=False) if isinstance(result, (dict, list)) else str(result)
            if len(content) > 25000:
                content = content[:25000] + "\n\n... [truncated due to size]"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": content
            })


# ====================== Usage example ======================
if __name__ == "__main__":
    report = review_branch(
        owner="your-org",
        repo="your-repo",
        base_branch="main",
        feature_branch="feature/new-functionality",
        # pr_number=123,          # uncomment if you have a PR number
        model="gpt-4o"
    )
    print(report)