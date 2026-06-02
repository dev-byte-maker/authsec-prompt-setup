"""Prompt Setup MCP Server — protected by AuthSec OAuth 2.0.

Implements the MCP Streamable-HTTP transport (stateless, one request per
HTTP round-trip).  AuthSec's mount_mcp middleware handles:

  • Bearer-token validation (JWKS + introspection)
  • tools/call scope enforcement (returns HTTP 403 on insufficient scope)
  • RFC 9728 protected-resource metadata at
    /.well-known/oauth-protected-resource/mcp

This file handles:
  • tools/list filtered by the caller's granted scopes
  • Prompt CRUD tool execution
  • Health endpoint for Railway
"""

from __future__ import annotations

import json
import logging
import os
import uuid

from dotenv import load_dotenv

load_dotenv()
from contextlib import asynccontextmanager
from typing import Any, Optional

from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from authsec_sdk import from_env, mount_mcp, principal_from_context
from authsec_sdk.runtime import PolicyMode, Runtime, ValidationMode
from authsec_sdk.runtime.manifest import ManifestTool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("prompt-setup")

# ─── AuthSec canonical scopes ─────────────────────────────────────────────────
_READ = frozenset({"prompt_setup:read", "prompt_setup:tools:read"})
_WRITE = frozenset({"prompt_setup:write", "prompt_setup:tools:write"})

# ─── Tool catalogue ────────────────────────────────────────────────────────────

_TOOL_DEFS: list[dict[str, Any]] = [
    {
        "name": "list_prompts",
        "description": "List all saved prompts.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_prompt",
        "description": "Retrieve a saved prompt by its ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Prompt ID"},
            },
            "required": ["id"],
        },
    },
    {
        "name": "create_prompt",
        "description": "Create and save a new prompt.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Short label for the prompt"},
                "content": {"type": "string", "description": "The prompt text"},
                "description": {"type": "string", "description": "Optional human-readable description"},
            },
            "required": ["name", "content"],
        },
    },
    {
        "name": "update_prompt",
        "description": "Update one or more fields of an existing prompt.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string"},
                "content": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["id"],
        },
    },
    {
        "name": "delete_prompt",
        "description": "Permanently delete a prompt by ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Prompt ID to delete"},
            },
            "required": ["id"],
        },
    },
]

# Caller needs ANY scope in the set to see / invoke the tool.
_TOOL_SCOPES: dict[str, frozenset[str]] = {
    "list_prompts":  _READ,
    "get_prompt":    _READ,
    "create_prompt": _WRITE,
    "update_prompt": _WRITE,
    "delete_prompt": _WRITE,
}


def _visible_tools(granted: frozenset[str]) -> list[dict[str, Any]]:
    """Return only the tools the caller is permitted to see."""
    return [
        t for t in _TOOL_DEFS
        if not _TOOL_SCOPES.get(t["name"]) or bool(_TOOL_SCOPES[t["name"]] & granted)
    ]


# ─── In-process prompt store ───────────────────────────────────────────────────
# Swap for a database in production.

_STORE: dict[str, dict[str, Any]] = {}


async def _list_prompts(_args: dict) -> Any:
    return list(_STORE.values())


async def _get_prompt(args: dict) -> Any:
    pid = args.get("id", "")
    if pid not in _STORE:
        raise KeyError(f"prompt {pid!r} not found")
    return _STORE[pid]


async def _create_prompt(args: dict) -> Any:
    pid = str(uuid.uuid4())
    entry: dict[str, Any] = {
        "id": pid,
        "name": args["name"],
        "content": args["content"],
        "description": args.get("description", ""),
    }
    _STORE[pid] = entry
    return entry


async def _update_prompt(args: dict) -> Any:
    pid = args.get("id", "")
    if pid not in _STORE:
        raise KeyError(f"prompt {pid!r} not found")
    entry = _STORE[pid]
    for field in ("name", "content", "description"):
        if field in args:
            entry[field] = args[field]
    return entry


async def _delete_prompt(args: dict) -> Any:
    pid = args.get("id", "")
    if pid not in _STORE:
        raise KeyError(f"prompt {pid!r} not found")
    del _STORE[pid]
    return {"deleted": pid}


_TOOL_FNS = {
    "list_prompts":  _list_prompts,
    "get_prompt":    _get_prompt,
    "create_prompt": _create_prompt,
    "update_prompt": _update_prompt,
    "delete_prompt": _delete_prompt,
}


# ─── MCP JSON-RPC helpers ──────────────────────────────────────────────────────

def _ok(msg_id: Any, result: Any) -> JSONResponse:
    return JSONResponse({"jsonrpc": "2.0", "id": msg_id, "result": result})


def _err(msg_id: Any, code: int, message: str) -> JSONResponse:
    return JSONResponse(
        {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}
    )


# ─── MCP Streamable-HTTP handler ──────────────────────────────────────────────

async def _mcp_handler(request: Request) -> Response:
    """
    Stateless MCP Streamable-HTTP endpoint.

    By the time execution reaches here, mount_mcp has already:
      1. Validated the bearer token and rejected invalid/inactive tokens.
      2. Blocked tools/call requests that lack the required scope.
      3. Set the AuthSec principal in the context var via principal_from_context().

    This function therefore only handles JSON-RPC dispatch.
    """
    if request.method == "GET":
        return JSONResponse({"server": "prompt-setup", "status": "ready"})

    body = await request.body()
    try:
        payload = json.loads(body) if body else {}
    except json.JSONDecodeError:
        return _err(None, -32700, "Parse error")

    msg_id = payload.get("id")
    method = payload.get("method", "")
    params = payload.get("params") or {}

    if method == "initialize":
        resp = _ok(msg_id, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "prompt-setup", "version": "1.0.0"},
        })
        resp.headers["Mcp-Session-Id"] = str(uuid.uuid4())
        return resp

    if method == "notifications/initialized":
        return Response(status_code=202)

    if method == "ping":
        return _ok(msg_id, {})

    if method == "tools/list":
        principal = principal_from_context()
        granted = frozenset(principal.scopes) if principal else frozenset()
        return _ok(msg_id, {"tools": _visible_tools(granted)})

    if method == "tools/call":
        # mount_mcp already rejected calls that lack scope; we just execute.
        tool_name = params.get("name", "")
        arguments = params.get("arguments") or {}
        fn = _TOOL_FNS.get(tool_name)
        if fn is None:
            return _err(msg_id, -32601, f"unknown tool {tool_name!r}")
        try:
            data = await fn(arguments)
            return _ok(msg_id, {
                "content": [{"type": "text", "text": json.dumps(data, indent=2)}],
                "isError": False,
            })
        except (KeyError, ValueError) as exc:
            return _ok(msg_id, {
                "content": [{"type": "text", "text": str(exc)}],
                "isError": True,
            })
        except Exception:
            logger.exception("tool %r raised an unexpected exception", tool_name)
            return _ok(msg_id, {
                "content": [{"type": "text", "text": "internal server error"}],
                "isError": True,
            })

    return _err(msg_id, -32601, f"method not found: {method!r}")


# ─── Health endpoint ───────────────────────────────────────────────────────────

async def _health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "server": "prompt-setup"})


# ─── AuthSec tool inventory (for manifest publishing) ─────────────────────────

def _tool_inventory() -> list[ManifestTool]:
    return [
        ManifestTool(
            name="list_prompts",
            description="List all saved prompts.",
            input_schema=_TOOL_DEFS[0]["inputSchema"],
            suggested_scopes=["prompt_setup:read"],
        ),
        ManifestTool(
            name="get_prompt",
            description="Retrieve a saved prompt by its ID.",
            input_schema=_TOOL_DEFS[1]["inputSchema"],
            suggested_scopes=["prompt_setup:read"],
        ),
        ManifestTool(
            name="create_prompt",
            description="Create and save a new prompt.",
            input_schema=_TOOL_DEFS[2]["inputSchema"],
            suggested_scopes=["prompt_setup:write"],
        ),
        ManifestTool(
            name="update_prompt",
            description="Update one or more fields of an existing prompt.",
            input_schema=_TOOL_DEFS[3]["inputSchema"],
            suggested_scopes=["prompt_setup:write"],
        ),
        ManifestTool(
            name="delete_prompt",
            description="Permanently delete a prompt by ID.",
            input_schema=_TOOL_DEFS[4]["inputSchema"],
            suggested_scopes=["prompt_setup:write"],
        ),
    ]


# ─── Application factory ───────────────────────────────────────────────────────

_runtime: Optional[Runtime] = None


@asynccontextmanager
async def lifespan(starlette_app: Starlette):
    # Starlette 1.x dropped on_event; call rt.startup() here instead.
    # mount_mcp's startup fetches the initial scope matrix and publishes the
    # tool manifest to AuthSec so admins can bind scopes in the UI.
    if _runtime is not None:
        await _runtime.startup()
    yield


def create_app() -> Starlette:
    global _runtime

    cfg = from_env()

    # Always enforce AuthSec canonical scopes regardless of env overrides.
    cfg.supported_scopes = [
        "prompt_setup:read",
        "prompt_setup:tools:read",
        "prompt_setup:tools:write",
        "prompt_setup:write",
    ]
    cfg.publish_manifest = True
    cfg.tool_inventory_provider = _tool_inventory
    cfg.tool_scope_suggestions = {
        "list_prompts":  ["prompt_setup:read"],
        "get_prompt":    ["prompt_setup:read"],
        "create_prompt": ["prompt_setup:write"],
        "update_prompt": ["prompt_setup:write"],
        "delete_prompt": ["prompt_setup:write"],
    }
    if cfg.policy_mode == PolicyMode.UNSET:
        cfg.policy_mode = PolicyMode.REMOTE_REQUIRED
    if cfg.validation_mode == ValidationMode.UNSET:
        cfg.validation_mode = ValidationMode.JWT_AND_INTROSPECT

    app = Starlette(
        lifespan=lifespan,
        routes=[Route("/health", _health, methods=["GET"])],
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Mcp-Session-Id", "Accept"],
        expose_headers=["Mcp-Session-Id"],
    )

    # mount_mcp registers:
    #   GET/POST /mcp                                    — auth middleware + MCP handler
    #   GET /.well-known/oauth-protected-resource/mcp   — RFC 9728 metadata
    _runtime = mount_mcp(app, "/mcp", _mcp_handler, cfg)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
