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

""""PanDA MCP client utilities.

This module provides:

* A factory function ``create_panda_mcp_client`` that returns a plain
  :class:`fastmcp.Client` configured like the original test client script.
* A convenience wrapper class :class:`PanDAMCPClient` for higher-level
  integrations (e.g., Ask PanDA's SelectionAgent).
* A simple async test function ``cl`` that can be used to verify
  connectivity and tool calls to the PanDA MCP server.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any, Dict, Optional, Sequence

from fastmcp import Client
from fastmcp.client.transports import SSETransport, StreamableHttpTransport


def _normalize_token(token: Optional[str]) -> Optional[str]:
    """Normalize a token string.

    Args:
        token: Raw token value, possibly ``None`` or an empty string.

    Returns:
        ``None`` if the input is ``None`` or empty/whitespace-only;
        otherwise, the stripped token string.
    """
    if token is None:
        return None
    stripped = token.strip()
    return stripped or None


def create_panda_mcp_client(
    host: str,
    port: int,
    *,
    transport: str,
    use_http: bool,
    token: Optional[str],
    vo: Optional[str],
) -> Client:
    """Create a configured PanDA MCP :class:`fastmcp.Client`.

    This mirrors the behavior of the original MCP test client:

    Example (original):

        if args.use_http:
            base_url = f"http://{args.host}:{args.port}/mcp/"
        else:
            base_url = f"https://{args.host}:{args.port}/mcp/"

        headers = {"Origin": args.vo} if args.token else None

        if args.transport == "streamable-http":
            transport = StreamableHttpTransport(
                url=base_url, auth=args.token, headers=headers
            )
        else:
            transport = SSETransport(url=base_url, auth=args.token, headers=headers)

    Args:
        host: Hostname of the PanDA MCP server.
        port: Port of the PanDA MCP server.
        transport: Transport type, either ``"streamable-http"`` or ``"sse"``.
        use_http: Whether to use HTTP (``True``) or HTTPS (``False``).
        token: Authentication token passed as ``auth`` to the transport.
        vo: Virtual organization string, used to set the ``Origin`` header.
            Only used if ``token`` is provided.

    Returns:
        A configured :class:`fastmcp.Client` instance.
    """
    """Create a configured PanDA MCP :class:`fastmcp.Client`."""

    # IMPORTANT: make empty strings behave like no token at all
    token = _normalize_token(token)

    if use_http:
        base_url = f"http://{host}:{port}/mcp/"
    else:
        base_url = f"https://{host}:{port}/mcp/"

    headers: Optional[Dict[str, str]] = {"Origin": vo} if token and vo else None

    if transport == "streamable-http":
        t = StreamableHttpTransport(url=base_url, auth=token, headers=headers)
    elif transport == "sse":
        t = SSETransport(url=base_url, auth=token, headers=headers)
    else:
        raise ValueError(f"Unsupported transport: {transport!r}")

    return Client(transport=t)

# ---------------------------------------------------------------------------
# High-level wrapper for Ask PanDA integration
# ---------------------------------------------------------------------------


class PanDAMCPClient:
    """High-level client wrapper for the PanDA MCP server.

    This wraps the configuration needed to talk to the PanDA MCP server and
    uses :func:`create_panda_mcp_client` internally for each operation.

    For health checks, it deliberately recreates a fresh low-level client and
    uses the same pattern as the working test script:

        client = create_panda_mcp_client(...)
        async with client:
            await client.call_tool("is_alive", {})

    This avoids subtle context-manager / reuse issues and ensures
    "Is the server alive?" style prompts follow the known-good code path.
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        transport: str,
        use_http: bool,
        token: Optional[str],
        vo: Optional[str],
    ) -> None:
        """Initialize a :class:`PanDAMCPClient` instance.

        Args:
            host: Hostname of the PanDA MCP server.
            port: Port of the PanDA MCP server.
            transport: Transport type, either ``"streamable-http"`` or ``"sse"``.
            use_http: Whether to use HTTP (``True``) or HTTPS (``False``).
            token: Authentication token passed as ``auth`` to the transport.
            vo: Virtual organization string, used to set the ``Origin`` header.
        """
        # Store config so we can recreate a low-level client whenever needed.
        self._host: str = host
        self._port: int = port
        self._transport: str = transport
        self._use_http: bool = use_http
        self._token: Optional[str] = token
        self._vo: Optional[str] = vo

        # Underlying client (used for generic calls if needed)
        self._client: Client = create_panda_mcp_client(
            host=host,
            port=port,
            transport=transport,
            use_http=use_http,
            token=token,
            vo=vo,
        )

    async def __aenter__(self) -> "PanDAMCPClient":
        """Enter the asynchronous context manager.

        Returns:
            The connected :class:`PanDAMCPClient` instance.
        """
        await self._client.connect()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[Any],
    ) -> None:
        """Exit the asynchronous context manager.

        Args:
            exc_type: Exception type, if any.
            exc: Exception instance, if any.
            tb: Traceback object, if any.
        """
        del exc_type, exc, tb  # unused
        await self._client.disconnect()

    @property
    def is_connected(self) -> bool:
        """bool: Whether the underlying client is currently connected."""
        return self._client.is_connected()

    async def list_tools(self) -> Sequence[Any]:
        """List tools exposed by the PanDA MCP server.

        Returns:
            A sequence of tool descriptors.
        """
        return await self._client.list_tools()

    async def call_tool(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Call a specific MCP tool on the PanDA MCP server.

        Args:
            tool_name: Name of the tool to invoke.
            arguments: Dictionary of arguments for the tool. If ``None``,
                an empty dictionary is used.

        Returns:
            Tool result payload as returned by the server.
        """
        return await self._client.call_tool(tool_name, arguments or {})

    async def is_server_running(self) -> bool:
        """Check whether the PanDA MCP server is reachable and responsive.

        Strategy:
            * Create a fresh :class:`fastmcp.Client` using the same
              configuration as the working test script.
            * Connect with ``async with client``.
            * Call the ``"is_alive"`` tool.
            * If that call succeeds without raising, consider the server
              "running".

        Returns:
            True if the server appears to be running and responsive,
            otherwise False.
        """
        try:
            client: Client = create_panda_mcp_client(
                host=self._host,
                port=self._port,
                transport=self._transport,
                use_http=self._use_http,
                token=self._token,
                vo=self._vo,
            )
            async with client:
                await client.call_tool("is_alive", {})
            return True
        except Exception:
            return False

    async def answer_prompt(self, prompt: str) -> str:
        """Answer a natural-language prompt using simple pattern routing.

        Currently supports questions about PanDA MCP server status, e.g.:

            "Is the server alive?"
            "Is the panda server alive?"
            "Is the PanDA server running?"
            "What is the PanDA MCP status?"

        Args:
            prompt: Natural-language input question.

        Returns:
            A human-readable answer string describing the server status or
            indicating that the prompt is not handled.
        """
        normalized = prompt.strip().lower()

        # Explicit phrases we care about
        server_keywords = (
            "is the panda server running",
            "is panda server running",
            "status of the panda server",
            "panda mcp status",
            "panda mcp server",
            "panda mcp health",
        )

        # Match explicit keywords OR generic "server + alive" pattern
        is_explicit = any(keyword in normalized for keyword in server_keywords)
        is_server_alive_like = ("server" in normalized and "alive" in normalized)

        if is_explicit or is_server_alive_like:
            running = await self.is_server_running()
            if running:
                return (
                    "Yes, the PanDA MCP server appears to be running and "
                    "responsive."
                )
            return (
                "No, the PanDA MCP server does not appear to be reachable or "
                "responsive right now."
            )

        return (
            "PanDAMCPClient received a prompt it does not currently handle: "
            f"{prompt!r}. Consider adding a dedicated handler for this case."
        )

    def answer_prompt_sync(self, prompt: str) -> str:
        """Synchronously answer a natural-language prompt.

        Args:
            prompt: Natural-language input question.

        Returns:
            A human-readable answer string.
        """
        return asyncio.run(self.answer_prompt(prompt))

    def is_server_running_sync(self) -> bool:
        """Synchronously check whether the PanDA MCP server is running.

        Returns:
            True if the server appears to be running and responsive,
            otherwise False.
        """
        return asyncio.run(self.is_server_running())

# ---------------------------------------------------------------------------
# CLI / test harness
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CLI interface.

    Returns:
        Configured :class:`argparse.ArgumentParser` instance.
    """
    parser = argparse.ArgumentParser(description="PanDA MCP client")

    parser.add_argument(
        "--host",
        default="localhost",
        type=str,
        help="PanDA MCP server host.",
    )
    parser.add_argument(
        "--port",
        default=8000,
        type=int,
        help="PanDA MCP server port.",
    )
    parser.add_argument(
        "--use-http",
        action="store_true",
        help="Use HTTP instead of HTTPS (default is HTTPS if not set).",
    )
    parser.add_argument(
        "--transport",
        default="streamable-http",
        choices=["streamable-http", "sse"],
        help="MCP transport type.",
    )
    parser.add_argument(
        "--token",
        default=None,
        type=str,
        help="Authentication token for the MCP server.",
    )
    parser.add_argument(
        "--vo",
        default=None,
        type=str,
        help="VO string used for Origin header (required by some servers).",
    )

    parser.add_argument(
        "--tool",
        default=None,
        type=str,
        help="Optional: name of a specific MCP tool to test (e.g. 'panda_health').",
    )
    parser.add_argument(
        "--kv",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Optional: key=value pairs for the tool arguments.",
    )

    parser.add_argument(
        "--question",
        default=None,
        type=str,
        help="Natural-language question to send via PanDAMCPClient.answer_prompt.",
    )
    parser.add_argument(
        "--model",
        default=None,
        type=str,
        help="Unused here, but kept for compatibility with other clients.",
    )

    return parser


def parse_kv_pairs(pairs: Sequence[str]) -> Dict[str, Any]:
    """Parse key=value pairs from the CLI into a dictionary.

    Args:
        pairs: Sequence of strings in the form ``KEY=VALUE``.

    Returns:
        Dictionary mapping keys to values (all values are strings).
    """
    result: Dict[str, Any] = {}
    for item in pairs:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        result[key] = value
    return result


async def cl(args: argparse.Namespace) -> None:
    """Main async entry point for the CLI.

    Behavior:

    * If ``--question`` is provided, use :class:`PanDAMCPClient` to
      answer a natural-language prompt (e.g. "Is the server alive?").
    * Otherwise, use the low-level :class:`fastmcp.Client` to:
        - connect,
        - list tools,
        - optionally call a specific tool if ``--tool`` is provided.

    Args:
        args: Parsed command-line arguments.
    """
    # First, handle the high-level question path (PanDAMCPClient wrapper).
    if args.question:
        wrapper = PanDAMCPClient(
            host=args.host,
            port=args.port,
            transport=args.transport,
            use_http=args.use_http,
            token=args.token,
            vo=args.vo,
        )
        answer = await wrapper.answer_prompt(args.question)
        print(answer)
        return

    # Otherwise, run in "raw test" mode with a direct fastmcp.Client.
    client: Client = create_panda_mcp_client(
        host=args.host,
        port=args.port,
        transport=args.transport,
        use_http=args.use_http,
        token=args.token,
        vo=args.vo,
    )

    async with client:
        if client.is_connected():
            print("Client connected")
        else:
            print("Client failed to connect")
            return

        tools = await client.list_tools()
        print("\nAvailable tools:")
        for tool in tools:
            name = getattr(tool, "name", "<unknown>")
            description = getattr(tool, "description", "")
            print(f"- {name} -")
            if description:
                print(f"  Description: {description}")
            print()

        if args.tool:
            kv_args = parse_kv_pairs(args.kv)
            print("\n" * 2)
            print(f"Testing tool {args.tool!r} with args {kv_args!r}:")
            result = await client.call_tool(args.tool, kv_args)
            print(f"Result: {result}\n")

    if not client.is_connected():
        print("Client disconnected")
    else:
        print("Client still connected")
    print("Done")


def main() -> None:
    """Entry point for ``python -m clients.panda_mcp``."""
    parser = build_arg_parser()
    args = parser.parse_args()
    asyncio.run(cl(args))


if __name__ == "__main__":
    main()
