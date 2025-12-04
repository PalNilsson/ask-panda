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
# Authors:
# - Paul Nilsson, paul.nilsson@cern.ch, 2025

"""Selection agent for Ask PanDA.

This module implements a SelectionAgent that:
- Classifies incoming questions into high-level categories (document, task, log analysis, etc.).
- Routes questions to the appropriate client/agent.
- Integrates with a generic PanDA MCP client (PanDAMCPClient) that discovers tools dynamically.

The key design goal is to avoid hardcoding any MCP tool names in the client/agent code.
Instead, the PanDA MCP server exposes tools and their docstrings via MCP; the LLM uses
that metadata to decide which tool to call and with what arguments.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import requests

from clients.document_query import DocumentQuery
from clients.log_analysis import LogAnalysis
from clients.data_query import TaskStatus
from clients.panda_mcp import PanDAMCPClient

# ---------------------------------------------------------------------------
# Module-level logger and defaults
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv("ASK_PANDA_LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)

ASK_PANDA_BASE_URL = os.getenv("ASK_PANDA_BASE_URL", "http://127.0.0.1:8000")


# ---------------------------------------------------------------------------
# Utility: Client bundle
# ---------------------------------------------------------------------------

@dataclass
class ClientBundle:
    """Container for all Ask PanDA clients.

    Attributes:
        document: Client for static documentation / RAG.
        queue: Placeholder for future queue client.
        task: Client for task status queries.
        log_analyzer: Client for log analysis.
        pilot_activity: Placeholder for pilot monitoring client.
        panda_mcp: Generic PanDA MCP client.
    """

    document: Optional[DocumentQuery]
    queue: Optional[Any]
    task: Optional[TaskStatus]
    log_analyzer: Optional[LogAnalysis]
    pilot_activity: Optional[Any]
    panda_mcp: Optional[PanDAMCPClient]


def get_clients(
    model: str,
    session_id: Optional[str],
    pandaid: Optional[str],
    taskid: Optional[str],
    cache: str,
) -> ClientBundle:
    """Create and return a bundle of clients for the different categories.

    Args:
        model: LLM model identifier.
        session_id: Optional session ID for conversation context.
        pandaid: Optional PanDA job ID for log analysis.
        taskid: Optional PanDA task ID for task status.
        cache: Path or key for cache storage.

    Returns:
        ClientBundle: Container object with initialized clients.
    """
    document_client = DocumentQuery(model, session_id)
    queue_client = None  # Placeholder for future queue client.

    task_client = TaskStatus(model, taskid, cache, session_id) if session_id and taskid else None
    log_client = LogAnalysis(model, pandaid, cache, session_id) if pandaid else None
    pilot_client = None  # Placeholder for a future PilotMonitorAgent, etc.

    # Generic MCP client using environment variables (PANDA_MCP_BASE_URL, etc.)
    panda_mcp: Optional[PanDAMCPClient]
    try:
        panda_mcp = PanDAMCPClient.from_env()
    except ValueError as exc:
        logger.warning("PanDA MCP client not configured: %s", exc)
        panda_mcp = None

    return ClientBundle(
        document=document_client,
        queue=queue_client,
        task=task_client,
        log_analyzer=log_client,
        pilot_activity=pilot_client,
        panda_mcp=panda_mcp,
    )


# ---------------------------------------------------------------------------
# Selection agent
# ---------------------------------------------------------------------------

class Selection:
    """Selection agent to route questions to appropriate Ask PanDA clients.

    The Selection agent has two main responsibilities:

    1. Classify an incoming natural-language question into a high-level category
       (e.g., "document", "task", "log_analyzer", "panda_mcp").
    2. Based on the category, forward the question to the appropriate client and
       return the resulting answer.

    For the PanDA MCP category, the Selection agent uses a generic PanDAMCPClient
    that discovers tools dynamically from the MCP server. It does not hardcode
    any specific tool names. Instead, it queries the MCP server for tool metadata
    and uses an LLM to choose which tool to call and with which arguments.
    """

    def __init__(
        self,
        clients: ClientBundle,
        model: str,
        session_id: Optional[str],
    ) -> None:
        """Initialize a Selection instance.

        Args:
            clients: A bundle of Ask PanDA clients for different categories.
            model: LLM model identifier to use when calling the Ask PanDA server.
            session_id: Optional session ID for conversation context.
        """
        self.clients = clients
        self.model = model
        self.session_id = session_id

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def simple_classification(self, question: str) -> str:
        """Classify the question into a category using heuristic rules.

        This is a lightweight, heuristic-based classifier. It can be replaced
        or augmented by an LLM-based classifier if needed.

        Args:
            question: Natural-language question from the user.

        Returns:
            Category string, one of:
            - "document"
            - "queue"
            - "task"
            - "log_analyzer"
            - "pilot_activity"
            - "panda_mcp"
            - "undefined"
        """
        q_lower = question.lower()

        # 1) Explicit MCP mentions always go to panda_mcp.
        if "panda mcp" in q_lower or "mcp" in q_lower:
            return "panda_mcp"

        # 2) Health / status questions about the PanDA server.
        health_words = ("alive", "health", "status", "up", "reachable", "responsive")
        if "panda" in q_lower and any(w in q_lower for w in health_words):
            # Examples:
            #   "Is the panda server alive?"
            #   "Is panda up?"
            #   "Check the status of the panda server"
            return "panda_mcp"

        # 3) Log analysis.
        if any(word in q_lower for word in ("log", "stderr", "stdout", "stack trace", "traceback")):
            return "log_analyzer"

        # 4) Task-related.
        if "task" in q_lower or "tasks" in q_lower:
            return "task"

        # 5) Queue / site-related.
        if "queue" in q_lower or "site" in q_lower:
            return "queue"

        # 6) Pilot activity.
        if "pilot" in q_lower:
            return "pilot_activity"

        # 7) Generic PanDA / ATLAS / documentation questions.
        if any(word in q_lower for word in ("atlas", "panda", "documentation", "manual", "how does")):
            return "document"

        # 8) Fallback.
        return "undefined"

    # ------------------------------------------------------------------
    # Main ask / routing
    # ------------------------------------------------------------------

    def ask(self, question: str) -> str:
        """Answer a user question by routing it to the appropriate client.

        Args:
            question: Natural-language question from the user.

        Returns:
            Answer as a string. If routing or downstream processing fails,
            an error message is returned instead.
        """
        category = self.simple_classification(question)
        logger.info("Initial classification: %s", category)

        if category == "document" and self.clients.document:
            logger.info("Routing question to DocumentQuery client")
            return self.clients.document.ask(question)

        if category == "task" and self.clients.task:
            logger.info("Routing question to TaskStatus client")
            return self.clients.task.ask(question)

        if category == "log_analyzer" and self.clients.log_analyzer:
            logger.info("Routing question to LogAnalysis client")
            return self.clients.log_analyzer.ask(question)

        if category == "pilot_activity" and self.clients.pilot_activity:
            logger.info("Routing question to PilotMonitor client")
            return self.clients.pilot_activity.ask(question)

        if category == "panda_mcp" and self.clients.panda_mcp:
            logger.info("Routing question to PanDAMCPClient with MCP-based tool selection")
            return self._route_panda_mcp_question(question, self.clients.panda_mcp)

        # Fallback: treat as a document / RAG question if possible.
        if self.clients.document:
            logger.info(
                "Category '%s' is undefined or missing client. "
                "Falling back to DocumentQuery.",
                category,
            )
            return self.clients.document.ask(question)

        return (
            "Error: Unable to route the question due to missing clients. "
            "Please check the Ask PanDA server configuration."
        )

    def _format_mcp_result(self, tool_name: str, result: Any) -> str:
        """Format a PanDA MCP tool result into a user-friendly string.

        This method adds a small UX layer on top of the raw JSON returned by
        MCP tools. It is intentionally conservative: if it does not recognize
        the structure, it falls back to pretty-printed JSON or a string
        representation.

        Args:
            tool_name: Name of the MCP tool that was called.
            result: Raw result returned from ``PanDAMCPClient.call_tool()``.

        Returns:
            Human-readable string representation of the result.
        """
        # Common PanDA pattern: {"success": bool, "message": str, "data": ...}
        if isinstance(result, dict):
            success = result.get("success")
            message = result.get("message")
            data = result.get("data")

            # Special-case health-style tools like is_alive
            if tool_name == "is_alive" and isinstance(success, bool):
                if success:
                    return "PanDA MCP reports: the service is **alive** and reachable."
                return "PanDA MCP reports: the service is **not** alive or returned an error."

            # Generic handling for this common shape
            if isinstance(success, bool) and ("message" in result or "data" in result):
                status = "succeeded" if success else "failed"
                parts = [f"PanDA MCP tool '{tool_name}' {status}."]
                if isinstance(message, str) and message.strip():
                    parts.append(f"Message: {message.strip()}")
                # If data is non-trivial, append a JSON dump
                if data not in (None, "", [], {}):
                    try:
                        data_json = json.dumps(data, indent=2, sort_keys=True)
                        parts.append("Data:\n" + data_json)
                    except Exception:  # noqa: BLE001
                        parts.append(f"Data: {data!r}")
                return "\n".join(parts)

            # Fallback for arbitrary dicts: pretty-printed JSON
            try:
                return json.dumps(result, indent=2, sort_keys=True)
            except Exception:  # noqa: BLE001
                return str(result)

        # List result: pretty-print JSON if possible
        if isinstance(result, list):
            try:
                return json.dumps(result, indent=2, sort_keys=True)
            except Exception:  # noqa: BLE001
                return str(result)

        # All other types: string representation
        return str(result)

    # ------------------------------------------------------------------
    # PanDA MCP routing
    # ------------------------------------------------------------------

    def _route_panda_mcp_question(
        self,
        question: str,
        client: PanDAMCPClient,
    ) -> str:
        """Route a PanDA MCP question via the generic PanDAMCPClient.

        This routing method follows four steps:

        1. Connect to the PanDA MCP server and obtain the list of tools,
           including their descriptions and JSON schemas, using
           ``client.describe_tools_for_llm()``.
        2. Send a routing prompt to the Ask PanDA server, instructing the LLM
           to choose a single tool and construct the arguments to call it.
        3. Parse the LLM's JSON response to extract the tool name and arguments.
        4. Call the chosen tool with ``client.call_tool(tool_name, arguments)``
           and return the formatted result.

        The design explicitly avoids hardcoding any tool names. The MCP server
        is the source of truth for available tools, and the LLM uses the tool
        metadata (names, docstrings, parameter schemas) to select the tool.

        Args:
            question: Natural-language user question about PanDA or its APIs.
            client: Initialized PanDAMCPClient instance.

        Returns:
            Answer string derived from the MCP tool's result or an error message.
        """

        async def _inner() -> str:
            """Async implementation of the MCP routing logic.

            Returns:
                Answer string or error message.
            """
            async with client:
                tools_description = await client.describe_tools_for_llm()

                router_prompt = f"""
You are a routing assistant for the PanDA MCP (Model Context Protocol) server.

You are given:
- A list of available MCP tools, including their descriptions and JSON schemas.
- A user question about PanDA, its APIs, or related operations.

Your task is to:
1. Carefully inspect the tools and their parameters.
2. Decide which single tool is best suited to answer the user's question.
3. Construct a valid JSON object of arguments for that tool, matching its JSON schema.

Output format:
Return ONLY a JSON object with exactly the following keys:

{{
  "tool_name": "<tool name as a string>",
  "arguments": {{ ... JSON object of arguments ... }}
}}

Do NOT include any additional text or explanation.

Tools and schemas:
{tools_description}

User question:
{question}
"""

                server_url = os.getenv("ASK_PANDA_BASE_URL", f"{ASK_PANDA_BASE_URL}/rag_ask")
                try:
                    response = requests.post(
                        server_url,
                        json={"question": router_prompt, "model": self.model},
                        timeout=120,
                    )
                except requests.RequestException as exc:
                    logger.error("Error contacting Ask PanDA server for MCP routing: %s", exc)
                    return (
                        "Error: could not contact the Ask PanDA server for MCP routing. "
                        f"Details: {exc}"
                    )

                if not response.ok:
                    logger.error(
                        "Ask PanDA MCP router returned HTTP %s: %s",
                        response.status_code,
                        response.text,
                    )
                    return (
                        "Error: Ask PanDA MCP router returned an HTTP error "
                        f"{response.status_code}. Please try again later."
                    )

                # Some Ask PanDA deployments return {"answer": "..."} JSON; others
                # may return plain text. Try both.
                try:
                    payload = response.json()
                    router_answer = payload.get("answer", payload)
                except Exception:  # noqa: BLE001
                    router_answer = response.text

                if isinstance(router_answer, dict):
                    routing_obj = router_answer
                else:
                    raw_text = str(router_answer).strip()

                    # Many LLMs wrap JSON in ```json ... ``` fences. Strip them if present.
                    if raw_text.startswith("```"):
                        # Remove opening fence (``` or ```json)
                        lines = raw_text.splitlines()
                        if lines and lines[0].lstrip().startswith("```"):
                            lines = lines[1:]
                        # Remove closing fence if present
                        if lines and lines[-1].strip().startswith("```"):
                            lines = lines[:-1]
                        raw_text = "\n".join(lines).strip()

                    try:
                        routing_obj = json.loads(raw_text)
                    except Exception as exc:  # noqa: BLE001
                        logger.error(
                            "Failed to parse routing JSON from LLM: %s; content=%r",
                            exc,
                            router_answer,
                        )
                        return (
                            "Error: could not interpret the PanDA MCP tool selection. "
                            "Please refine your question or specify the API operation you need."
                        )

                if not isinstance(routing_obj, Mapping):
                    logger.error("Routing output is not a JSON object: %r", routing_obj)
                    return (
                        "Error: unexpected format in PanDA MCP routing output. "
                        "Please try rephrasing your question."
                    )

                tool_name = routing_obj.get("tool_name")
                arguments = routing_obj.get("arguments", {})

                if not isinstance(tool_name, str):
                    logger.error("Routing result missing 'tool_name': %r", routing_obj)
                    return (
                        "Error: routing result did not specify a tool_name. "
                        "Please refine your question."
                    )

                if not isinstance(arguments, dict):
                    logger.error("Routing result 'arguments' is not a dict: %r", routing_obj)
                    return (
                        "Error: routing result contained invalid arguments format. "
                        "Please refine your question."
                    )

                logger.info(
                    "PanDA MCP routing selected tool '%s' with arguments %r",
                    tool_name,
                    arguments,
                )

                try:
                    result = await client.call_tool(tool_name, arguments)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Error calling MCP tool %s: %s", tool_name, exc)
                    return (
                        f"Error while calling PanDA MCP tool '{tool_name}': {exc}. "
                        "Please check the tool name and arguments."
                    )

                # Normalize the result to a human-readable string.
                return self._format_mcp_result(tool_name, result)

        # Run the async logic synchronously for CLI usage.
        return asyncio.run(_inner())


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Selection agent.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Ask PanDA Selection Agent")
    parser.add_argument(
        "--question",
        "-q",
        required=True,
        help="User question to route and answer.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("ASK_PANDA_MODEL", "gemini-2.5-flash"),
        help="LLM model identifier (default: env ASK_PANDA_MODEL or gemini-2.5-flash).",
    )
    parser.add_argument(
        "--session-id",
        "--session_id",
        dest="session_id",
        default=None,
        help="Optional session ID for conversation context.",
    )
    parser.add_argument(
        "--pandaid",
        default=None,
        help="Optional PanDA job ID for log analysis.",
    )
    parser.add_argument(
        "--taskid",
        default=None,
        help="Optional PanDA task ID for task status queries.",
    )
    parser.add_argument(
        "--cache",
        default=os.getenv("ASK_PANDA_CACHE", "/tmp/ask_panda_cache"),
        help="Cache path or key for Ask PanDA clients.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the Selection agent CLI.

    This function:

    1. Parses command-line arguments.
    2. Instantiates the required Ask PanDA clients.
    3. Creates a ``Selection`` instance.
    4. Routes the question and prints the answer.

    The presence of the PanDA MCP client is optional. If it is not configured
    (for example, if ``PANDA_MCP_BASE_URL`` is not set), questions that would
    otherwise route to PanDA MCP fall back to document/RAG handling.
    """
    args = parse_args()
    clients = get_clients(
        model=args.model,
        session_id=args.session_id,
        pandaid=args.pandaid,
        taskid=args.taskid,
        cache=args.cache,
    )

    selection_agent = Selection(clients, args.model, args.session_id)
    logger.info("Received query: %r", args.question)

    answer = selection_agent.ask(args.question)
    print(answer)


if __name__ == "__main__":
    main()
