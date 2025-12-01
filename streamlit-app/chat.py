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

import sys
from io import StringIO
import contextlib
import importlib
import uuid
from typing import Dict, Any

import streamlit as st
import clients.selection as selection  # your existing client


def run_selection_client(
    question: str,
    model: str,
    session_id: str | None = None,
    cache_dir: str | None = None,
):
    """
    Call clients.selection.main() as if run from the command line.

    We temporarily override sys.argv so that argparse inside
    clients.selection sees the expected arguments.
    """
    argv = ["clients.selection", "--question", question, "--model", model]

    if session_id:
        argv.extend(["--session-id", session_id])

    if cache_dir:
        argv.extend(["--cache", cache_dir])

    old_argv = sys.argv
    stdout_buf = StringIO()
    stderr_buf = StringIO()

    # Reload in case you edit clients/selection.py while Streamlit is running
    importlib.reload(selection)

    try:
        sys.argv = argv

        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(
            stderr_buf
        ):
            try:
                answer = selection.main()
                exit_code = 0
            except SystemExit as e:
                exit_code = e.code if isinstance(e.code, int) else 1
                answer = None
    finally:
        sys.argv = old_argv

    stdout = stdout_buf.getvalue()
    stderr = stderr_buf.getvalue()

    return answer, stdout, stderr, exit_code


# ---------------- Chat state helpers ---------------- #

def create_new_chat(title: str | None = None) -> Dict[str, Any]:
    """Create a new chat object with its own session_id and greeting."""
    chat_id = uuid.uuid4().hex
    session_id = f"streamlit-{chat_id[:8]}"
    return {
        "id": chat_id,
        "title": title or "New chat",
        "session_id": session_id,
        "messages": [
            {
                "role": "assistant",
                "content": (
                    "Hello, I am Ask PanDA's Streamlit interface. "
                    "Ask me about PanDA tasks, jobs, queues, and more!"
                ),
            }
        ],
        "last_stdout": "",
        "last_stderr": "",
    }


def ensure_chat_state():
    """Initialize chats list and current_chat_index on first load."""
    if "chats" not in st.session_state or not st.session_state.chats:
        st.session_state.chats = [create_new_chat(title="Chat 1")]
        st.session_state.current_chat_index = 0

    if "current_chat_index" not in st.session_state:
        st.session_state.current_chat_index = 0

    # Safety: clamp index if needed
    if st.session_state.current_chat_index >= len(st.session_state.chats):
        st.session_state.current_chat_index = len(st.session_state.chats) - 1


def get_current_chat() -> Dict[str, Any]:
    return st.session_state.chats[st.session_state.current_chat_index]


def set_current_chat(index: int):
    st.session_state.current_chat_index = index


def update_chat_title_from_prompt(chat: Dict[str, Any], prompt: str):
    """If the chat still has a generic title, derive one from the first user prompt."""
    if chat["title"] in ("New chat", "Chat 1", ""):
        trimmed = prompt.strip().replace("\n", " ")
        if len(trimmed) > 40:
            trimmed = trimmed[:37] + "..."
        chat["title"] = trimmed or "New chat"


# ---------------- Streamlit UI ---------------- #

st.set_page_config(
    page_title="Ask PanDA",
    page_icon="🐼",
    layout="wide",
)

# ---- ChatGPT-like styling for the page and chat input ----
CHATGPT_CSS = """
<style>

/* White main background */
[data-testid="stAppViewContainer"] {
    background: white !important;
}

/* ChatGPT-like centered main column */
section[data-testid="stMain"] > div {
    max-width: 900px;
    margin: 0 auto;
}

/* Input container spacing */
div[data-testid="stChatInput"] {
    padding-top: 12px;
    padding-bottom: 24px;
}

/* Chat input wrapper */
div[data-testid="stChatInput"] > div {
    border-radius: 18px !important;
    border: 1px solid rgba(0, 0, 0, 0.2);
    background: #f2f2f2 !important;  /* light gray */
}

/* The textarea itself */
div[data-testid="stChatInput"] textarea {
    min-height: 75px !important;      /* adjust to your liking */
    max-height: 250px !important;
    padding: 26px 18px !important;
    display: flex;
    align-items: center;
    font-size: 1rem !important;
    line-height: 1.4 !important;
    resize: none !important;
    border-radius: 12px !important;
    border: none !important;
    background: #f2f2f2 !important;   /* light gray */
    color: #000 !important;
}

div[data-testid="stChatInput"] > div:focus-within {
    outline: none !important;
    box-shadow: none !important;
    border-color: rgba(0,0,0,0.2) !important;  /* keep your neutral border */
}
/* Placeholder color */
div[data-testid="stChatInput"] textarea::placeholder {
    color: #666 !important;
}

/* Button tweaks */
div[data-testid="stChatInput"] button[kind="primary"] {
    border-radius: 999px !important;
}

</style>
"""

st.markdown(CHATGPT_CSS, unsafe_allow_html=True)

st.title("🐼 Ask PanDA")

# Initialize multi-chat state
ensure_chat_state()
current_chat = get_current_chat()

# ---- Read initial question from URL (?q=...) ----
try:
    # Newer Streamlit: st.query_params is a mapping-like object
    qp = st.query_params
except AttributeError:
    # Older versions: use experimental_get_query_params
    qp = st.experimental_get_query_params()

# qp may be a dict or a QueryParams object; both support .get("q")
raw_q = qp.get("q")
if isinstance(raw_q, list):
    initial_q = raw_q[0]
else:
    initial_q = raw_q

# We'll use this for the very first turn if present
# Only do this once per chat
if initial_q:
    key = f"initial_q_processed_{current_chat['id']}"
    if not st.session_state.get(key, False):
        # Mark as processed early to avoid loops
        st.session_state[key] = True

        # Default model/cache for the auto-first-turn; user can change later
        default_model = st.session_state.get("default_model", "gemini")
        default_cache_dir = "cache"

        # Add user message
        current_chat["messages"].append({"role": "user", "content": initial_q})
        update_chat_title_from_prompt(current_chat, initial_q)

        # Call Ask PanDA once on load
        with st.spinner("Querying Ask PanDA via clients.selection…"):
            answer, stdout, stderr, exit_code = run_selection_client(
                question=initial_q.strip(),
                model=default_model,
                session_id=current_chat["session_id"],
                cache_dir=default_cache_dir,
            )

        current_chat["last_stdout"] = stdout
        current_chat["last_stderr"] = stderr

        if exit_code != 0 or answer is None:
            error_msg = (
                f"❌ The Ask PanDA client exited with code **{exit_code}**.\n\n"
                "Check the logs in the sidebar (stderr) for details.\n\n"
                "Most common causes:\n"
                "- MCP server not running or not reachable.\n"
                "- Misconfigured base URL / network issues.\n"
                "- Internal exception in one of the sub-clients."
            )
            current_chat["messages"].append(
                {"role": "assistant", "content": error_msg}
            )
        else:
            if isinstance(answer, dict):
                if "answer" in answer:
                    content = str(answer["answer"])
                else:
                    content = f"```json\n{answer}\n```"
            else:
                content = str(answer)

            current_chat["messages"].append(
                {"role": "assistant", "content": content}
            )

        # Rerun so the full conversation is rendered properly
        st.rerun()


# ---- Sidebar: chat history + settings ----
with st.sidebar:
    st.header("Chats")

    # Show chat list as a radio group
    chat_titles = [chat["title"] for chat in st.session_state.chats]
    selected_index = st.radio(
        "Chat history",
        options=list(range(len(chat_titles))),
        format_func=lambda i: chat_titles[i],
        index=st.session_state.current_chat_index,
        key="chat_selector",
    )

    if selected_index != st.session_state.current_chat_index:
        set_current_chat(selected_index)
        st.rerun()

    # New chat button: add a chat and switch to it
    if st.button("🆕 New chat", use_container_width=True):
        st.session_state.chats.append(create_new_chat())
        st.session_state.current_chat_index = len(st.session_state.chats) - 1
        st.rerun()

    st.markdown("---")
    st.header("Settings")

    # Model to pass to --model
    model = st.text_input(
        "Model",
        value="gemini",
        help="Model name passed to --model (same as in your CLI).",
        key="model_input",
    )
    st.session_state.default_model = model

    # Show and allow editing of the session_id for the current chat
    session_id = st.text_input(
        "Session ID",
        value=current_chat["session_id"],
        help="Used by ContextMemory to keep conversation history.",
        key=f"session_id_input_{current_chat['id']}",
    )
    current_chat["session_id"] = session_id

    cache_dir = st.text_input(
        "Cache directory",
        value="cache",
        help="Value passed to --cache (optional).",
        key="cache_input",
    )

    st.markdown("---")
    st.subheader("Last client logs (current chat)")
    last_stdout = current_chat.get("last_stdout", "")
    last_stderr = current_chat.get("last_stderr", "")

    with st.expander("stdout"):
        if last_stdout.strip():
            st.code(last_stdout, language="text")
        else:
            st.caption("No stdout captured yet for this chat.")

    with st.expander("stderr"):
        if last_stderr.strip():
            st.code(last_stderr, language="text")
        else:
            st.caption("No stderr captured yet for this chat.")


# ---- Render chat history for the current chat ----
for msg in current_chat["messages"]:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])


# ---- Chat input (bottom) ----
prompt = st.chat_input("Send a message...")

if prompt:
    if not model.strip():
        st.warning("Please specify a model in the sidebar.")
        st.stop()

    # 1) Add user message to current chat
    current_chat["messages"].append({"role": "user", "content": prompt})
    update_chat_title_from_prompt(current_chat, prompt)

    with st.chat_message("user"):
        st.markdown(prompt)

    # 2) Call clients.selection and add assistant reply
    with st.chat_message("assistant"):
        with st.spinner("Querying Ask PanDA via clients.selection…"):
            answer, stdout, stderr, exit_code = run_selection_client(
                question=prompt.strip(),
                model=model.strip(),
                session_id=current_chat["session_id"],
                cache_dir=cache_dir.strip() or None,
            )

        # Store logs in this chat
        current_chat["last_stdout"] = stdout
        current_chat["last_stderr"] = stderr

        if exit_code != 0 or answer is None:
            error_msg = (
                f"❌ The Ask PanDA client exited with code **{exit_code}**.\n\n"
                "Check the logs in the sidebar (stderr) for details.\n\n"
                "Most common causes:\n"
                "- MCP server not running or not reachable.\n"
                "- Misconfigured base URL / network issues.\n"
                "- Internal exception in one of the sub-clients."
            )
            st.markdown(error_msg)
            current_chat["messages"].append(
                {"role": "assistant", "content": error_msg}
            )
        else:
            if isinstance(answer, dict):
                if "answer" in answer:
                    content = str(answer["answer"])
                    st.markdown(content)
                else:
                    content = f"```json\n{answer}\n```"
                    st.markdown(content)
            else:
                content = str(answer)
                st.markdown(content)

            current_chat["messages"].append(
                {"role": "assistant", "content": content}
            )

    # Rerun to redraw the full conversation with updated history, title, logs
    st.rerun()
