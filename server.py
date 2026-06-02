"""
AuthSec-protected GitHub MCP Server.

Serves GitHub tools over MCP's JSON-RPC HTTP transport, protected by AuthSec
OAuth 2.0 bearer token validation and scope-matrix enforcement.

Bootstrap path: from_env() → Config → mount_mcp(app, "/mcp", handler, cfg)
The SDK registers:
  POST /mcp                                               — bearer-protected MCP endpoint
  GET  /.well-known/oauth-protected-resource/mcp         — RFC 9728 protected-resource metadata

Authoritative scopes (AuthSec canonical; no legacy scopes):
  mcp:read, mcp:tools:read  — read-only GitHub operations
  mcp:write, mcp:tools:write — mutating GitHub operations

GitHub credentials (GITHUB_TOKEN) are server-side only and never forwarded
to callers or reflected in AuthSec tokens.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from dotenv import load_dotenv
from typing import Any, Optional

import httpx
from contextlib import asynccontextmanager
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from authsec_sdk import ManifestTool, from_env
from authsec_sdk.runtime import Config, PolicyMode, ValidationMode
from authsec_sdk.runtime.server import (
    InsufficientScopeError,
    PolicyUnavailableError,
    Runtime,
)
from authsec_sdk.runtime.metadata import (
    build_resource_metadata_path,
    metadata_json_response,
    build_www_authenticate,
)
from authsec_sdk.runtime.validator import TokenInvalidError, TokenInactiveError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
_LOG = logging.getLogger("mcp.server")

# Load environment variables from a .env file when present (development convenience)
load_dotenv()

# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

GITHUB_API = "https://api.github.com"

# AuthSec canonical scopes — the ONLY scopes this server recognises.
# Do NOT add: read, write, admin, tools.read, tools.write, openid, profile,
#             email, offline_access, default, all, or *.
_READ_SCOPES:  list[str] = ["mcp:read", "mcp:tools:read"]
_WRITE_SCOPES: list[str] = ["mcp:write", "mcp:tools:write"]

TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_repos",
        "description": "List repositories for a GitHub user or organization.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "GitHub user or org login"},
                "type": {
                    "type": "string",
                    "enum": ["all", "owner", "member"],
                    "default": "owner",
                },
                "per_page": {"type": "integer", "default": 30, "maximum": 100},
            },
            "required": ["owner"],
        },
    },
    {
        "name": "get_repo",
        "description": "Get details about a specific GitHub repository.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
            },
            "required": ["owner", "repo"],
        },
    },
    {
        "name": "list_issues",
        "description": "List issues in a GitHub repository.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "state": {
                    "type": "string",
                    "enum": ["open", "closed", "all"],
                    "default": "open",
                },
                "labels": {
                    "type": "string",
                    "description": "Comma-separated label names to filter by",
                },
                "per_page": {"type": "integer", "default": 30, "maximum": 100},
            },
            "required": ["owner", "repo"],
        },
    },
    {
        "name": "get_issue",
        "description": "Get a specific issue from a GitHub repository.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "issue_number": {"type": "integer"},
            },
            "required": ["owner", "repo", "issue_number"],
        },
    },
    {
        "name": "create_issue",
        "description": "Create a new issue in a GitHub repository.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "title": {"type": "string"},
                "body": {"type": "string", "description": "Issue body in Markdown"},
                "labels": {"type": "array", "items": {"type": "string"}},
                "assignees": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["owner", "repo", "title"],
        },
    },
    {
        "name": "list_pull_requests",
        "description": "List pull requests in a GitHub repository.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "state": {
                    "type": "string",
                    "enum": ["open", "closed", "all"],
                    "default": "open",
                },
                "per_page": {"type": "integer", "default": 30, "maximum": 100},
            },
            "required": ["owner", "repo"],
        },
    },
    {
        "name": "create_pull_request",
        "description": "Open a new pull request in a GitHub repository.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "title": {"type": "string"},
                "body": {
                    "type": "string",
                    "description": "PR description in Markdown",
                },
                "head": {
                    "type": "string",
                    "description": "Branch containing the changes",
                },
                "base": {
                    "type": "string",
                    "description": "Branch to merge changes into",
                },
                "draft": {"type": "boolean", "default": False},
            },
            "required": ["owner", "repo", "title", "head", "base"],
        },
    },
    {
        "name": "get_file_contents",
        "description": "Get the contents of a file from a GitHub repository.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "path": {
                    "type": "string",
                    "description": "File path within the repository",
                },
                "ref": {
                    "type": "string",
                    "description": "Branch, tag, or commit SHA (defaults to default branch)",
                },
            },
            "required": ["owner", "repo", "path"],
        },
    },
    {
        "name": "push_files",
        "description": "Create or update files in a GitHub repository via a single commit.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "branch": {
                    "type": "string",
                    "description": "Target branch for the commit",
                },
                "message": {"type": "string", "description": "Commit message"},
                "files": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string", "description": "UTF-8 text content"},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            "required": ["owner", "repo", "branch", "message", "files"],
        },
    },
    {
        "name": "search_code",
        "description": "Search for code across GitHub repositories.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "GitHub code search query (supports qualifiers like repo:, lang:)",
                },
                "per_page": {"type": "integer", "default": 10, "maximum": 30},
            },
            "required": ["query"],
        },
    },
]

# Local fallback scope map — AuthSec canonical scopes ONLY.
# Consulted when the remote scope matrix is temporarily unreachable.
LOCAL_TOOL_SCOPES: dict[str, list[str]] = {
    "list_repos":          _READ_SCOPES,
    "get_repo":            _READ_SCOPES,
    "list_issues":         _READ_SCOPES,
    "get_issue":           _READ_SCOPES,
    "list_pull_requests":  _READ_SCOPES,
    "get_file_contents":   _READ_SCOPES,
    "search_code":         _READ_SCOPES,
    "create_issue":        _WRITE_SCOPES,
    "create_pull_request": _WRITE_SCOPES,
    "push_files":          _WRITE_SCOPES,
}

# Scope hints published to the AuthSec manifest — same values as local fallback.
TOOL_SCOPE_SUGGESTIONS: dict[str, list[str]] = dict(LOCAL_TOOL_SCOPES)


# ---------------------------------------------------------------------------
# GitHub HTTP helpers  (server-side credentials only)
# ---------------------------------------------------------------------------

def _gh_headers() -> dict[str, str]:
    """Build GitHub API headers using the server-side PAT.

    The token is read from the environment on every call so a secret rotation
    (without restart) takes effect immediately.  It is NEVER forwarded to
    callers or included in any AuthSec principal.
    """
    headers: dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "authsec-mcp-server/1.0",
    }
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

async def _call_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    try:
        text = await _dispatch(name, args)
        return {"content": [{"type": "text", "text": text}]}
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"GitHub API error {exc.response.status_code}: {exc.response.text[:512]}"
        ) from exc


async def _dispatch(name: str, args: dict[str, Any]) -> str:  # noqa: C901 (long but intentional)
    async with httpx.AsyncClient(headers=_gh_headers(), timeout=30.0) as gh:

        if name == "list_repos":
            owner = args["owner"]
            r = await gh.get(
                f"{GITHUB_API}/users/{owner}/repos",
                params={"type": args.get("type", "owner"), "per_page": args.get("per_page", 30)},
            )
            r.raise_for_status()
            return json.dumps(
                [
                    {
                        "name": repo["name"],
                        "full_name": repo["full_name"],
                        "description": repo.get("description"),
                        "private": repo["private"],
                        "stars": repo["stargazers_count"],
                        "default_branch": repo["default_branch"],
                        "url": repo["html_url"],
                    }
                    for repo in r.json()
                ],
                indent=2,
            )

        if name == "get_repo":
            r = await gh.get(
                f"{GITHUB_API}/repos/{args['owner']}/{args['repo']}"
            )
            r.raise_for_status()
            repo = r.json()
            return json.dumps(
                {
                    "full_name": repo["full_name"],
                    "description": repo.get("description"),
                    "private": repo["private"],
                    "stars": repo["stargazers_count"],
                    "forks": repo["forks_count"],
                    "open_issues": repo["open_issues_count"],
                    "default_branch": repo["default_branch"],
                    "url": repo["html_url"],
                    "created_at": repo["created_at"],
                    "updated_at": repo["updated_at"],
                },
                indent=2,
            )

        if name == "list_issues":
            params: dict[str, Any] = {
                "state": args.get("state", "open"),
                "per_page": args.get("per_page", 30),
            }
            if args.get("labels"):
                params["labels"] = args["labels"]
            r = await gh.get(
                f"{GITHUB_API}/repos/{args['owner']}/{args['repo']}/issues",
                params=params,
            )
            r.raise_for_status()
            return json.dumps(
                [
                    {
                        "number": issue["number"],
                        "title": issue["title"],
                        "state": issue["state"],
                        "user": issue["user"]["login"],
                        "labels": [lb["name"] for lb in issue["labels"]],
                        "created_at": issue["created_at"],
                        "url": issue["html_url"],
                    }
                    for issue in r.json()
                    if "pull_request" not in issue  # exclude PRs that appear in issues API
                ],
                indent=2,
            )

        if name == "get_issue":
            r = await gh.get(
                f"{GITHUB_API}/repos/{args['owner']}/{args['repo']}"
                f"/issues/{args['issue_number']}"
            )
            r.raise_for_status()
            issue = r.json()
            return json.dumps(
                {
                    "number": issue["number"],
                    "title": issue["title"],
                    "body": issue.get("body", ""),
                    "state": issue["state"],
                    "user": issue["user"]["login"],
                    "labels": [lb["name"] for lb in issue["labels"]],
                    "assignees": [a["login"] for a in issue["assignees"]],
                    "created_at": issue["created_at"],
                    "updated_at": issue["updated_at"],
                    "url": issue["html_url"],
                },
                indent=2,
            )

        if name == "create_issue":
            body: dict[str, Any] = {"title": args["title"]}
            if args.get("body"):
                body["body"] = args["body"]
            if args.get("labels"):
                body["labels"] = args["labels"]
            if args.get("assignees"):
                body["assignees"] = args["assignees"]
            r = await gh.post(
                f"{GITHUB_API}/repos/{args['owner']}/{args['repo']}/issues",
                json=body,
            )
            r.raise_for_status()
            issue = r.json()
            return json.dumps(
                {"number": issue["number"], "url": issue["html_url"], "title": issue["title"]},
                indent=2,
            )

        if name == "list_pull_requests":
            r = await gh.get(
                f"{GITHUB_API}/repos/{args['owner']}/{args['repo']}/pulls",
                params={
                    "state": args.get("state", "open"),
                    "per_page": args.get("per_page", 30),
                },
            )
            r.raise_for_status()
            return json.dumps(
                [
                    {
                        "number": pr["number"],
                        "title": pr["title"],
                        "state": pr["state"],
                        "user": pr["user"]["login"],
                        "head": pr["head"]["ref"],
                        "base": pr["base"]["ref"],
                        "draft": pr.get("draft", False),
                        "created_at": pr["created_at"],
                        "url": pr["html_url"],
                    }
                    for pr in r.json()
                ],
                indent=2,
            )

        if name == "create_pull_request":
            body = {
                "title": args["title"],
                "head": args["head"],
                "base": args["base"],
                "draft": args.get("draft", False),
            }
            if args.get("body"):
                body["body"] = args["body"]
            r = await gh.post(
                f"{GITHUB_API}/repos/{args['owner']}/{args['repo']}/pulls",
                json=body,
            )
            r.raise_for_status()
            pr = r.json()
            return json.dumps(
                {"number": pr["number"], "url": pr["html_url"], "title": pr["title"]},
                indent=2,
            )

        if name == "get_file_contents":
            params = {}
            if args.get("ref"):
                params["ref"] = args["ref"]
            r = await gh.get(
                f"{GITHUB_API}/repos/{args['owner']}/{args['repo']}"
                f"/contents/{args['path']}",
                params=params,
            )
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list):
                # directory listing
                return json.dumps(
                    [{"name": f["name"], "type": f["type"], "size": f["size"]} for f in data],
                    indent=2,
                )
            content_b64 = data.get("content", "")
            return base64.b64decode(content_b64.replace("\n", "")).decode(
                "utf-8", errors="replace"
            )

        if name == "push_files":
            owner = args["owner"]
            repo = args["repo"]
            branch = args["branch"]

            # Resolve branch tip
            r = await gh.get(
                f"{GITHUB_API}/repos/{owner}/{repo}/git/ref/heads/{branch}"
            )
            r.raise_for_status()
            commit_sha = r.json()["object"]["sha"]

            # Resolve base tree
            r = await gh.get(
                f"{GITHUB_API}/repos/{owner}/{repo}/git/commits/{commit_sha}"
            )
            r.raise_for_status()
            base_tree_sha = r.json()["tree"]["sha"]

            # Create blobs
            tree_items: list[dict[str, Any]] = []
            for f in args["files"]:
                br = await gh.post(
                    f"{GITHUB_API}/repos/{owner}/{repo}/git/blobs",
                    json={"content": f["content"], "encoding": "utf-8"},
                )
                br.raise_for_status()
                tree_items.append(
                    {
                        "path": f["path"],
                        "mode": "100644",
                        "type": "blob",
                        "sha": br.json()["sha"],
                    }
                )

            # Create tree
            tr = await gh.post(
                f"{GITHUB_API}/repos/{owner}/{repo}/git/trees",
                json={"base_tree": base_tree_sha, "tree": tree_items},
            )
            tr.raise_for_status()
            new_tree_sha = tr.json()["sha"]

            # Create commit
            cr = await gh.post(
                f"{GITHUB_API}/repos/{owner}/{repo}/git/commits",
                json={
                    "message": args["message"],
                    "tree": new_tree_sha,
                    "parents": [commit_sha],
                },
            )
            cr.raise_for_status()
            new_commit_sha = cr.json()["sha"]

            # Advance branch ref
            rr = await gh.patch(
                f"{GITHUB_API}/repos/{owner}/{repo}/git/refs/heads/{branch}",
                json={"sha": new_commit_sha},
            )
            rr.raise_for_status()

            return json.dumps(
                {
                    "commit": new_commit_sha,
                    "branch": branch,
                    "files": [f["path"] for f in args["files"]],
                    "url": f"https://github.com/{owner}/{repo}/commit/{new_commit_sha}",
                },
                indent=2,
            )

        if name == "search_code":
            r = await gh.get(
                f"{GITHUB_API}/search/code",
                params={"q": args["query"], "per_page": args.get("per_page", 10)},
            )
            r.raise_for_status()
            data = r.json()
            return json.dumps(
                {
                    "total_count": data["total_count"],
                    "items": [
                        {
                            "name": item["name"],
                            "path": item["path"],
                            "repository": item["repository"]["full_name"],
                            "url": item["html_url"],
                        }
                        for item in data["items"]
                    ],
                },
                indent=2,
            )

        raise ValueError(f"Unknown tool: {name!r}")


# ---------------------------------------------------------------------------
# MCP protocol helpers
# ---------------------------------------------------------------------------

_SERVER_INFO: dict[str, Any] = {
    "protocolVersion": "2024-11-05",
    "serverInfo": {"name": "authsec-github-mcp", "version": "1.0.0"},
    "capabilities": {"tools": {}},
}


def _rpc_ok(req_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _rpc_err(req_id: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": err}


# ---------------------------------------------------------------------------
# In-process RPC handler — used by the AuthSec manifest publisher only.
# No bearer auth is applied here; this path never reaches external clients.
# ---------------------------------------------------------------------------

async def _manifest_rpc_handler(
    request: dict[str, Any], headers: dict[str, str]
) -> dict[str, Any]:
    method = request.get("method", "")
    req_id = request.get("id")

    if method == "initialize":
        return _rpc_ok(req_id, _SERVER_INFO)
    if method in ("notifications/initialized",):
        return {}  # Notifications have no response
    if method == "tools/list":
        return _rpc_ok(req_id, {"tools": TOOLS})
    return _rpc_err(req_id, -32601, "Method not found")


# ---------------------------------------------------------------------------
# Inventory provider — passed to Config so the SDK can publish the manifest
# ---------------------------------------------------------------------------

def _inventory_provider() -> list[ManifestTool]:
    return [
        ManifestTool(
            name=t["name"],
            description=t["description"],
            input_schema=t["inputSchema"],
            suggested_scopes=TOOL_SCOPE_SUGGESTIONS.get(t["name"], []),
        )
        for t in TOOLS
    ]


# ---------------------------------------------------------------------------
# tools/list scope filter
# ---------------------------------------------------------------------------

async def _visible_tools(principal: Any) -> list[dict[str, Any]]:
    """Return the subset of TOOLS the bearer token is authorised to call.

    When AuthSec is in OPEN mode (no resource_server_id) every tool is visible.
    When the runtime is absent (dev mode) every tool is visible.
    """
    if _runtime is None:
        return list(TOOLS)

    visible: list[dict[str, Any]] = []
    for tool in TOOLS:
        try:
            await _runtime.authorize_tool(principal, tool["name"])
            visible.append(tool)
        except (InsufficientScopeError, PolicyUnavailableError):
            pass
    return visible


# ---------------------------------------------------------------------------
# HTTP MCP handler  (registered via mount_mcp — auth already verified)
# ---------------------------------------------------------------------------

async def mcp_handler(request: Request) -> Response:
    principal = getattr(request.state, "authsec_principal", None)

    try:
        raw = await request.body()
        payload: dict[str, Any] = json.loads(raw) if raw else {}
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JSONResponse(_rpc_err(None, -32700, "Parse error"), status_code=400)

    method = payload.get("method", "")
    req_id = payload.get("id")

    if method == "initialize":
        return JSONResponse(_rpc_ok(req_id, _SERVER_INFO))

    if method == "notifications/initialized":
        return Response(status_code=202)

    if method == "tools/list":
        tools = await _visible_tools(principal)
        return JSONResponse(_rpc_ok(req_id, {"tools": tools}))

    if method == "tools/call":
        params = payload.get("params") or {}
        name = params.get("name", "")
        args: dict[str, Any] = params.get("arguments") or {}
        try:
            result = await _call_tool(name, args)
            return JSONResponse(_rpc_ok(req_id, result))
        except ValueError as exc:
            return JSONResponse(_rpc_err(req_id, -32602, str(exc)), status_code=400)
        except RuntimeError as exc:
            return JSONResponse(
                _rpc_err(req_id, -32000, "Tool execution failed", str(exc)),
                status_code=502,
            )

    return JSONResponse(_rpc_err(req_id, -32601, "Method not found"), status_code=404)


# ---------------------------------------------------------------------------
# AuthSec configuration
# ---------------------------------------------------------------------------

# Hardcoded AuthSec service endpoints — never change unless the tenant moves.
_AUTHSEC_ISSUER           = "https://dev.api.authsec.dev"
_AUTHSEC_JWKS_URL         = "https://dev.api.authsec.dev/oauth/jwks"
_AUTHSEC_INTROSPECTION_URL = "https://dev.api.authsec.dev/oauth/introspect"
_RESOURCE_URI             = "https://authsec-mcp-production.up.railway.app/mcp"
_RESOURCE_NAME            = "mcp"
_CANONICAL_SCOPES         = ["mcp:read", "mcp:tools:read", "mcp:tools:write", "mcp:write"]


def _build_config() -> Config:
    """Build the AuthSec Config, preferring env vars over hardcoded defaults."""
    cfg = from_env()

    # Apply known-good defaults for fields that from_env() left empty.
    if not cfg.issuer:
        cfg.issuer = _AUTHSEC_ISSUER
    if not cfg.authorization_server:
        cfg.authorization_server = _AUTHSEC_ISSUER
    if not cfg.jwks_url:
        cfg.jwks_url = _AUTHSEC_JWKS_URL
    if not cfg.introspection_url:
        cfg.introspection_url = _AUTHSEC_INTROSPECTION_URL
    if not cfg.resource_uri:
        cfg.resource_uri = _RESOURCE_URI
    if not cfg.resource_name:
        cfg.resource_name = _RESOURCE_NAME

    # Always advertise only the canonical scope set.
    cfg.supported_scopes = _CANONICAL_SCOPES

    # Local fallback: AuthSec canonical scopes only (no legacy scope strings).
    cfg.tool_scopes = LOCAL_TOOL_SCOPES
    cfg.tool_scope_suggestions = TOOL_SCOPE_SUGGESTIONS

    # Always use REMOTE_WITH_LOCAL_FALLBACK when a resource_server_id is configured.
    #
    # REMOTE_REQUIRED blocks ALL tool calls (raises PolicyUnavailableError) while
    # the remote scope matrix has not yet reached policy_complete=true — i.e. during
    # the bootstrap phase before the admin has mapped tools to scopes in the AuthSec
    # dashboard. This makes tools/list return [] and breaks the "Discovery snapshot"
    # and "tools/list filter" validations.
    #
    # REMOTE_WITH_LOCAL_FALLBACK lets the server serve tools via the local scope map
    # (LOCAL_TOOL_SCOPES) while the remote matrix is being configured. Once the admin
    # activates the resource server in AuthSec, the SDK picks up the remote policy on
    # the next TTL refresh (≤30 s) and local scopes become the fallback-only path.
    if cfg.resource_server_id:
        cfg.policy_mode = PolicyMode.REMOTE_WITH_LOCAL_FALLBACK
    else:
        _LOG.warning(
            "AUTHSEC_RESOURCE_SERVER_ID not set — scope matrix unavailable; "
            "using LOCAL_ONLY policy (local tool_scopes apply)"
        )
        cfg.policy_mode = PolicyMode.LOCAL_ONLY

    if cfg.validation_mode == ValidationMode.UNSET:
        cfg.validation_mode = ValidationMode.JWT_AND_INTROSPECT

    # Push the tool manifest to AuthSec at startup so the admin UI reflects
    # the current tool set without a manual re-registration.
    cfg.publish_manifest = True
    cfg.tool_inventory_provider = _inventory_provider

    return cfg


# ---------------------------------------------------------------------------
# AuthSec route handlers — defined before app construction so routes can be
# passed to Starlette() directly instead of appended after construction.
# ---------------------------------------------------------------------------

_runtime: Optional[Runtime] = None
_cfg: Optional[Config] = None


async def _authsec_metadata(request: Request) -> Response:
    """GET /.well-known/oauth-protected-resource/mcp — RFC 9728 metadata."""
    if _runtime is None:
        return JSONResponse({"error": "service_unavailable"}, status_code=503)
    authoritative = await _runtime.get_authoritative_scopes()
    body, resp_headers = metadata_json_response(_runtime.cfg, authoritative)
    return Response(content=body, media_type="application/json", headers=resp_headers)


async def _authsec_mcp(request: Request) -> Response:
    """POST /mcp — bearer-token validation + scope enforcement, then dispatch."""
    if _runtime is None:
        return JSONResponse({"error": "service_unavailable"}, status_code=503)

    cfg = _runtime.cfg

    # ── 1. Extract bearer token ───────────────────────────────────────────
    auth_header = request.headers.get("authorization", "")
    parts = auth_header.split(None, 1)
    token = parts[1].strip() if len(parts) == 2 and parts[0].lower() == "bearer" else ""

    if not token:
        www = build_www_authenticate(cfg, error="invalid_token",
                                     error_description="missing bearer token")
        return JSONResponse(
            {"error": "invalid_token", "error_description": "missing bearer token"},
            status_code=401,
            headers={"WWW-Authenticate": www},
        )

    # ── 2. Validate token ─────────────────────────────────────────────────
    try:
        principal = await _runtime.validate_token(token)
    except (TokenInvalidError, TokenInactiveError) as exc:
        www = build_www_authenticate(cfg, error="invalid_token",
                                     error_description=str(exc))
        return JSONResponse(
            {"error": "invalid_token", "error_description": str(exc)},
            status_code=401,
            headers={"WWW-Authenticate": www},
        )
    except Exception:
        _LOG.exception("unexpected token validation failure")
        www = build_www_authenticate(cfg, error="invalid_token",
                                     error_description="token validation failure")
        return JSONResponse(
            {"error": "invalid_token", "error_description": "token validation failure"},
            status_code=401,
            headers={"WWW-Authenticate": www},
        )

    request.state.authsec_principal = principal

    # ── 3. tools/call scope enforcement ───────────────────────────────────
    body_bytes = await request.body()
    if body_bytes and request.method == "POST":
        try:
            payload = json.loads(body_bytes)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and payload.get("method") == "tools/call":
            tool_name = (payload.get("params") or {}).get("name", "")
            if tool_name:
                try:
                    await _runtime.authorize_tool(principal, tool_name)
                except InsufficientScopeError as exc:
                    scope = " ".join(exc.required)
                    www = build_www_authenticate(cfg, error="insufficient_scope",
                                                 error_description=f"tool {exc.tool!r} requires {exc.required!r}",
                                                 scope=scope)
                    return JSONResponse(
                        {"error": "insufficient_scope",
                         "error_description": str(exc),
                         "tool": exc.tool,
                         "required_scopes": exc.required},
                        status_code=403,
                        headers={"WWW-Authenticate": www},
                    )
                except PolicyUnavailableError as exc:
                    return JSONResponse(
                        {"error": "policy_unavailable",
                         "error_description": str(exc)},
                        status_code=503,
                    )

    # ── 4. Replay consumed body and dispatch to MCP handler ───────────────
    sent = False

    async def _replay():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body_bytes, "more_body": False}
        return {"type": "http.disconnect"}

    request._receive = _replay  # type: ignore[attr-defined]
    return await mcp_handler(request)


# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

_AUTHSEC_ENABLED = os.environ.get("AUTHSEC_ENABLED", "true").lower() not in ("false", "0", "no")


async def _startup() -> None:
    if _runtime is not None:
        await _runtime.startup(rpc_handler=_manifest_rpc_handler)


@asynccontextmanager
async def _lifespan(app):
    asyncio.create_task(_startup())
    yield


async def _health(request: Request) -> Response:
    """Railway healthcheck — always 200 so the service is never marked unhealthy."""
    return JSONResponse({"status": "ok"})


# Compute the metadata path now (before app construction) using the env var
# that was loaded by load_dotenv() above.
_meta_path = build_resource_metadata_path(
    os.environ.get("AUTHSEC_RESOURCE_URI", _RESOURCE_URI)
)

# Build route list at construction time — never append routes after Starlette()
# is created, as that is unreliable in some ASGI deployment environments.
if _AUTHSEC_ENABLED:
    try:
        _cfg = _build_config()
        _runtime = Runtime(_cfg)
        _meta_path = build_resource_metadata_path(_cfg.resource_uri)
        _LOG.info(
            "AuthSec protection active — resource_uri=%s policy_mode=%s",
            _cfg.resource_uri,
            _cfg.effective_policy_mode().value,
        )
    except Exception as exc:
        _LOG.error(
            "AuthSec initialization failed (%s: %s) — check AUTHSEC_* env vars. "
            "Metadata endpoint will return 503 until credentials are configured.",
            type(exc).__name__, exc,
        )

    _routes = [
        Route("/health", _health, methods=["GET"]),
        Route(_meta_path, _authsec_metadata, methods=["GET"]),
        Route("/mcp", _authsec_mcp if _runtime is not None else mcp_handler,
              methods=["GET", "POST"]),
    ]
else:
    _LOG.warning("AUTHSEC_ENABLED=false — running WITHOUT token validation.")
    _routes = [
        Route("/health", _health, methods=["GET"]),
        Route("/mcp", mcp_handler, methods=["GET", "POST"]),
    ]

app = Starlette(routes=_routes, lifespan=_lifespan)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        log_level="info",
    )
