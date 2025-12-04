# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# Authors:
# - Paul Nilsson, paul.nilsson@cern.ch, 2025

"""
Generic PanDA MCP client for Ask PanDA.

This module provides a thin, general-purpose wrapper around the PanDA MCP
server using fastmcp.Client. It does *not* hardcode any specific tools;
instead, it exposes generic methods to:

- connect to the PanDA MCP endpoint
- discover available tools (names, descriptions, parameter schemas)
- call arbitrary tools by name with structured arguments

Intended usage:
    from clients.panda_mcp import PanDAMCPClient

    async def main():
        async with PanDAMCPClient.from_env() as client:
            # 1. Discover tools
            tools = client.list_tools()
            for t in tools:
                print(t.name, "-", t.description)

            # 2. Call a tool dynamically
            result = await client.call_tool("is_alive", {})
            print(result)

You can then feed `client.list_tools()` into your LLM as context so it can
choose which tool to call based on descriptions and argument schemas.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from fastmcp import Client as MCPClient
from fastmcp.client.transports import SSETransport, StreamableHttpTransport


@dataclass
class PanDAMCPConfig:
    """Configuration for connecting to a PanDA MCP server.

    This configuration supports two styles of environment variables:

    New style (preferred):
        - PANDA_MCP_BASE_URL  (e.g. "http://host:25080/mcp/")

    Legacy style (backward compatible):
        - PANDA_MCP_HOST      (e.g. "pandaserver01.sdcc.bnl.gov")
        - PANDA_MCP_PORT      (default: "25080")
        - PANDA_MCP_PATH      (default: "/mcp/")
        - PANDA_MCP_USE_HTTP  ("true"/"false"; if "false", use https)

    Common extras (both styles):
        - PANDA_MCP_USE_SSE   ("1"/"true"/"yes" to use SSE)
        - PANDA_MCP_TOKEN     (bearer token, optional)
        - PANDA_MCP_ORIGIN    (Origin header value, optional)
        - PANDA_MCP_TIMEOUT   (float seconds, optional)

    Attributes:
        base_url: Base URL of the MCP HTTP endpoint.
        use_sse: Whether to use SSETransport.
        auth_token: Optional bearer token for authentication.
        origin: Optional Origin header value.
        timeout: Optional request timeout in seconds.
    """

    base_url: str
    use_sse: bool = False
    auth_token: Optional[str] = None
    origin: Optional[str] = None
    timeout: Optional[float] = None

    @classmethod
    def from_env(cls) -> "PanDAMCPConfig":
        """Build configuration from environment variables.

        The method first checks PANDA_MCP_BASE_URL. If it is not set,
        it falls back to the legacy host/port/path variables.

        Returns:
            PanDAMCPConfig: Parsed configuration.

        Raises:
            ValueError: If neither PANDA_MCP_BASE_URL nor PANDA_MCP_HOST
                is set in the environment.
        """
        # 1) Try new-style base URL.
        base_url = os.environ.get("PANDA_MCP_BASE_URL")

        # 2) If not provided, build from legacy host/port/path.
        if not base_url:
            host = os.environ.get("PANDA_MCP_HOST")
            if not host:
                raise ValueError(
                    "PANDA_MCP_BASE_URL is not set and PANDA_MCP_HOST is not set. "
                    "Cannot construct PanDA MCP endpoint URL."
                )

            port = os.environ.get("PANDA_MCP_PORT", "25080")
            path = os.environ.get("PANDA_MCP_PATH", "/mcp/")

            # Decide scheme: if PANDA_MCP_USE_HTTP is explicitly "false",
            # assume https; otherwise default to http for backward compatibility.
            use_http_flag = os.environ.get("PANDA_MCP_USE_HTTP", "").lower()
            if use_http_flag == "false":
                scheme = "https"
            else:
                scheme = "http"

            # Ensure path starts and ends sensibly
            if not path.startswith("/"):
                path = "/" + path
            # FastMCP HTTP transport is fine whether or not we end with '/'
            base_url = f"{scheme}://{host}:{port}{path}"

        # 3) Common options.
        use_sse = os.environ.get("PANDA_MCP_USE_SSE", "").lower() in {"1", "true", "yes"}
        auth_token = os.environ.get("PANDA_MCP_TOKEN") or None
        origin = os.environ.get("PANDA_MCP_ORIGIN") or None
        timeout_str = os.environ.get("PANDA_MCP_TIMEOUT")
        timeout = float(timeout_str) if timeout_str else None

        return cls(
            base_url=base_url,
            use_sse=use_sse,
            auth_token=auth_token,
            origin=origin,
            timeout=timeout,
        )


class PanDAMCPClient:
    """
    Generic client for the PanDA MCP server.

    This is a thin wrapper around fastmcp.Client that focuses on:
    - tool discovery
    - generic tool invocation

    It deliberately avoids defining one method per tool. Instead, Ask PanDA
    agents (or an LLM) should:
        - inspect available tools via `list_tools()`
        - select the appropriate tool using the descriptions & schemas
        - call the tool via `call_tool(name, arguments)`

    Example:
        async with PanDAMCPClient.from_env() as client:
            # Discover tools
            tools = client.list_tools()
            for tool in tools:
                print(tool.name, tool.description)

            # Call the "is_alive" tool
            result = await client.call_tool("is_alive", {})
            print("Server says:", result)
    """

    def __init__(self, config: PanDAMCPConfig):
        self._config = config
        self._client: Optional[MCPClient] = None

    # ------------------------------------------------------------------
    # Construction / context management
    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls) -> "PanDAMCPClient":
        """
        Create a client using environment variables via PanDAMCPConfig.from_env().
        """
        config = PanDAMCPConfig.from_env()
        return cls(config)

    async def __aenter__(self) -> "PanDAMCPClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def connect(self) -> None:
        """
        Initialize the underlying fastmcp.Client and connect to the server.

        Normally you use this via `async with PanDAMCPClient(...)` so you
        rarely need to call `connect()` explicitly.
        """
        if self._client is not None:
            return  # already connected

        headers: Dict[str, str] = {}
        if self._config.auth_token:
            headers["Authorization"] = f"Bearer {self._config.auth_token}"
        if self._config.origin:
            headers["Origin"] = self._config.origin

        if self._config.use_sse:
            transport = SSETransport(
                url=self._config.base_url,
                headers=headers or None,
            )
        else:
            transport = StreamableHttpTransport(
                url=self._config.base_url,
                headers=headers or None,
            )

        self._client = MCPClient(
            name="ask-panda-mcp-client",
            transport=transport,
        )

        # Establish connection; handle different fastmcp versions.
        # Newer versions may expose `connect()`, older ones only `_connect`.
        if hasattr(self._client, "connect"):
            await self._client.connect()  # type: ignore[func-returns-value]
        elif hasattr(self._client, "_connect"):
            await self._client._connect()  # type: ignore[attr-defined]
        else:
            # Fallback: some fastmcp versions lazily connect on first use.
            # In that case, nothing to do here.
            pass

    async def close(self) -> None:
        """Close the underlying connection to the MCP server.

        This method handles different fastmcp versions:

        - Newer versions may expose an async ``aclose()`` method.
        - Older versions may only expose ``close()``.
        """
        if self._client is None:
            return

        # Newer fastmcp versions
        if hasattr(self._client, "aclose"):
            await self._client.aclose()  # type: ignore[func-returns-value]
        # Older fastmcp versions
        elif hasattr(self._client, "close"):
            await self._client.close()  # type: ignore[func-returns-value]

        self._client = None

    # ------------------------------------------------------------------
    # Tool discovery
    # ------------------------------------------------------------------
    def _require_client(self) -> MCPClient:
        if self._client is None:
            raise RuntimeError(
                "PanDAMCPClient is not connected. "
                "Use `async with PanDAMCPClient(...)` or await `.connect()` first."
            )
        return self._client

    async def list_tools(self) -> List[Any]:
        """Return the list of tools exposed by the PanDA MCP server.

        This method supports different fastmcp versions:

        - Some expose an async ``list_tools()`` method.
        - Others may expose a ``tools`` attribute (mapping or iterable).

        Returns:
            List of tool objects as returned by fastmcp.
        """
        client = self._require_client()

        # Preferred: async list_tools() method.
        if hasattr(client, "list_tools"):
            # Older or alternate fastmcp versions: list_tools() is async.
            tools = await client.list_tools()  # type: ignore[func-returns-value]
            return list(tools)

        # Fallback: tools attribute (if present in some versions).
        if hasattr(client, "tools"):
            tools_attr = getattr(client, "tools")
            if isinstance(tools_attr, Mapping):
                return list(tools_attr.values())
            return list(tools_attr)  # type: ignore[arg-type]

        # Last resort: no tool info available.
        return []

    def get_tool(self, name: str) -> Optional[Any]:
        """
        Retrieve a single tool by name, or None if no such tool exists.
        """
        client = self._require_client()
        if isinstance(client.tools, Mapping):
            return client.tools.get(name)
        # Fallback if tools is some other iterable type
        for tool in client.tools:
            if tool.name == name:
                return tool
        return None

    async def describe_tools_for_llm(self) -> str:
        """Return a human-readable description of all tools for LLM context.

        The returned string includes tool names, descriptions, and JSON schemas
        for their parameters. It is intended to be passed as context to an LLM
        so that it can decide which tool to use and how to construct arguments.

        Returns:
            Multi-line string describing all known tools.
        """
        tools = await self.list_tools()
        blocks: List[str] = []

        for t in tools:
            # Try to access name/description/input_schema in a robust way.
            name = getattr(t, "name", "<unknown>")
            description = getattr(t, "description", "") or "(no description)"
            schema = getattr(t, "input_schema", None)

            try:
                if hasattr(schema, "model_json_schema"):
                    schema_dict = schema.model_json_schema()
                elif hasattr(schema, "schema"):
                    schema_dict = schema.schema()
                else:
                    schema_dict = schema if schema is not None else {}
                schema_json = json.dumps(schema_dict, indent=2, sort_keys=True)
            except Exception:
                schema_json = "<unable to serialize schema>"

            block = (
                f"Tool name: {name}\n"
                f"Description:\n{description}\n"
                f"Parameters (JSON Schema):\n{schema_json}\n"
                "----"
            )
            blocks.append(block)

        return "\n\n".join(blocks)

    # ------------------------------------------------------------------
    # Tool invocation
    # ------------------------------------------------------------------
    async def call_tool(
            self,
            tool_name: str,
            arguments: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Call a specific PanDA MCP tool with the given arguments.

        Args:
            tool_name: Name of the tool to call (as advertised by MCP).
            arguments: Dict of arguments matching the tool's JSON schema.

        Returns:
            The tool's result in a convenient Python form:

            - If fastmcp returns a CallToolResult, this method preferentially
              returns its ``structured_content`` or ``data`` field (e.g., a dict).
            - Otherwise, it returns the raw result as-is.

        Raises:
            RuntimeError: If the client is not connected.
            Any underlying exception thrown by ``fastmcp.Client.call_tool()``.
        """
        client = self._require_client()
        result = await client.call_tool(tool_name, arguments or {})

        # fastmcp typically returns a CallToolResult object with several fields:
        #   - content (list of content blocks)
        #   - structured_content (dict/list/str)
        #   - data (dict/list/str)
        #   - is_error (bool)
        #
        # For Ask PanDA, we want the most structured representation.
        structured = getattr(result, "structured_content", None)
        if structured is not None:
            return structured

        data = getattr(result, "data", None)
        if data is not None:
            return data

        # Fallback: if a single text content exists, try to return that.
        content = getattr(result, "content", None)
        if isinstance(content, list) and content:
            first = content[0]
            text = getattr(first, "text", None)
            if isinstance(text, str):
                return text

        # Last resort: return the raw result.
        return result
