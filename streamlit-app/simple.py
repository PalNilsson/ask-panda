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

# streamlit-app/simple.py
import sys
from io import StringIO
import contextlib
import importlib

import streamlit as st
import clients.selection as selection  # <-- IMPORTANT: module-level import


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

    # Optional: reload in case you edit clients/selection.py while Streamlit is running
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


# ---------------- Streamlit UI ---------------- #

st.set_page_config(
    page_title="Ask PanDA",
    page_icon="🐼",
    layout="wide",
)

st.title("🐼 Ask PanDA – Streamlit Client")
st.markdown(
    "This UI calls the **clients.selection** client against the running MCP server."
)

with st.sidebar:
    st.header("Settings")

    model = st.text_input(
        "Model",
        value="gemini",
        help="Model name passed to --model (same as in your CLI)",
    )

    session_id = st.text_input(
        "Session ID",
        value="streamlit-session",
        help="Used by ContextMemory to keep conversation history.",
    )

    cache_dir = st.text_input(
        "Cache directory",
        value="cache",
        help="Value passed to --cache (optional).",
    )

    st.markdown("---")
    st.caption(
        "Make sure the Ask PanDA MCP server is running "
        "before using this UI."
    )

st.subheader("Ask a question")
question = st.text_area(
    "Your question",
    height=150,
    placeholder=(
        "e.g. Is the PanDA server alive?\n"
        "Or: What is the status of task 123456789?"
    ),
)

col1, col2 = st.columns([1, 4])
with col1:
    ask_clicked = st.button("Ask PanDA")

if ask_clicked:
    if not question.strip():
        st.warning("Please enter a question first.")
    elif not model.strip():
        st.warning("Please specify a model.")
    else:
        with st.spinner("Querying Ask PanDA via clients.selection..."):
            answer, stdout, stderr, exit_code = run_selection_client(
                question=question.strip(),
                model=model.strip(),
                session_id=session_id.strip() or None,
                cache_dir=cache_dir.strip() or None,
            )

        st.subheader("Answer")

        if exit_code != 0 or answer is None:
            st.error(
                f"clients.selection exited with code {exit_code}. "
                "Check the logs below for details."
            )
        else:
            if isinstance(answer, dict):
                if "answer" in answer:
                    st.markdown(answer["answer"])
                else:
                    st.json(answer)
            else:
                st.markdown(str(answer))

        with st.expander("Client logs (stdout)"):
            if stdout.strip():
                st.code(stdout, language="text")
            else:
                st.caption("No stdout captured.")

        with st.expander("Client logs (stderr)"):
            if stderr.strip():
                st.code(stderr, language="text")
            else:
                st.caption("No stderr captured.")
