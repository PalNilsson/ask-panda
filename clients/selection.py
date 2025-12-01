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

from logging_config import setup_logging
setup_logging("selection")
import logging
logger = logging.getLogger(__name__)

import argparse
import os
import re
import requests
import sys
from json import JSONDecodeError
from time import sleep

from clients.document_query import DocumentQuery
from clients.log_analysis import LogAnalysis
from clients.data_query import TaskStatus
from clients.panda_mcp import PanDAMCPClient
from clients.CRICanalysis import CRICanalysisClient, is_CRIC_related, schema_path
from tools.context_memory import ContextMemory
from tools.errorcodes import EC_TIMEOUT
from tools.server_utils import ASK_PANDA_BASE_URL, check_server_health

memory = ContextMemory()


class Selection:
    def __init__(self, clients: dict, model: str, session_id: str = None) -> None:
        self.clients = clients  # dict like {"document": ..., "queue": ...}
        self.model = model        # e.g., OpenAI or Anthropic wrapper
        self.session_id = session_id  # Session ID for tracking conversation

    def classify_question(self, question: str) -> str:
        """
        Classify the question into one of the predefined categories.

        This is where the rules are defined to classify the question to
        be able to select the appropriate client.

        Args:
            question (str): The question to classify.

        Returns:
            str: The category of the question (e.g., "document", "queue", "task", "log_analyzer", "pilot_activity", "CRIC_analyzer").
        """
        # no need to involve the LLM if we can classify it with simple rules
        initial = self.simple_classification(question)
        logger.info(f"Initial classification: {initial}")
        if initial != "undefined":
            return initial

        # if related to CRIC, return directly
        if is_CRIC_related(question):
            return "CRIC_analyzer"

        prompt = f"""
You are a routing assistant for a question-answering system. Your job is to classify a question into one of the following categories, based on its topic:

- document:
    Questions about general usage, concepts, how-to guides, or error messages that are answered from documentation
    or long-form knowledge of systems (e.g. PanDA, prun, pathena, containers, error codes).
- queue:
    Questions about site or queue data stored in a JSON file (e.g. queues at a site, corepower, copytool, status of a queue, which queues use rucio).
- log_analyzer:
    Questions about why a specific job failed (e.g. log or failure analysis of job NNN).
    The word 'job', 'pandaid' or 'panda id' must be followed by a number.
    If the last line in the question is a number and the user provides no other hint, you should assume it is a job id and select the 'job' category.
- task:
    Questions about a specific task's status or job counts (e.g. status of task NNN, number of failed jobs).
    The word 'task' or 'task id' or 'taskid' must be followed by a number.
    If the last line in the question is a number and the user provides no other hint, you should assume it is a task id and select the 'task' category.
- pilot_activity: Questions about pilot activity, failures, or statistics, possibly using information from Grafana
    (e.g. pilots running on queue X, pilots failing, links to Grafana).
- panda_mcp:
    Questions about whether the PanDA MCP server is alive, reachable or running
    (e.g. "Is the server alive?", "Is the PanDA server running?", "PanDA MCP status").

Classify the following question:

"{question}"

Output only one of the categories: document, queue, log_analyzer, task, pilot_activity, or panda_mcp.
"""
        result = self.ask(prompt, returnstring=True).strip().lower()
        return result if result in self.clients else "document"

    import re

    def simple_classification(self, question: str) -> str:
        """
        A simple rule-based classification of the question into categories.

        Args:
            question (str): The question to classify.

        Returns:
            str: The category of the question ("document", "task", "log_analyzer", "panda_mcp", or "undefined").
        """
        # Normalize text (case-insensitive matching)
        q = question.lower()

        # Quick detection of PanDA MCP / server-status questions
        panda_mcp_keywords = (
            "panda mcp",
            "panda mcp status",
            "panda mcp server",
            "panda server status",
            "is the server alive",
            "is the panda server running",
            "is panda server running",
        )
        if any(keyword in q for keyword in panda_mcp_keywords):
            return "panda_mcp"

        # Regex patterns for task and job with numbers
        task_pattern = r'\b(task|task id|taskid)\s*\d+'
        job_pattern = r'\b(job|job id|jobid|panda id|pandaid)\s*\d+'

        # Check for 'task' conditions
        if re.search(task_pattern, q):
            return "task"

        # Check for 'job' conditions
        if re.search(job_pattern, q):
            return "log_analyzer"

        # If not matched, check for both 'task' and 'job' words (not necessarily with numbers)
        if "task" in q or "job" in q:
            # if the last line is a "User: " followed by a number we cannot classify it here
            lines = q.strip().splitlines()
            if lines and re.match(r'^user:\s*\d+$', lines[-1].strip()):
                return "undefined"

            return "document"

        # Default case
        return "undefined"

    def answer(self, question: str) -> str:
        return self.classify_question(question)

    def ask(self, question: str, returnstring=False) -> str or dict:
        """
        Send a question to the LLM via the MCP server and retrieve the answer.

        This function sends the question and the selected model to the
        ASK_PANDA_BASE_URL server endpoint using a POST request. It expects a
        JSON response containing the answer. If the response indicates an error
        (with a non-200 status code), an error message is returned instead.

        Args:
            question (str): The question to ask the LLM.

        Returns:
            str or dict: The answer from the LLM, or a dictionary containing the session ID, if
            returnstring is False. If returnstring is True, returns the answer as a string.
        """
        server_url = os.getenv("ASK_PANDA_BASE_URL", f"{ASK_PANDA_BASE_URL}/rag_ask")

        # Construct prompt
        prompt = question

        try:
            response = requests.post(server_url, json={"question": prompt, "model": self.model}, timeout=30)
            if response.ok:
                try:
                    # Store interaction
                    _answer = response.json()["answer"]
                    if returnstring:
                        return _answer

                    answer = {
                        "session_id": self.session_id,
                        "question": question,
                        "model": self.model,
                        "answer": _answer
                    }
                    return answer
                except JSONDecodeError:  # Changed to use imported JSONDecodeError
                    return "Error: Could not decode JSON response from server."
                except KeyError:
                    return "Error: 'answer' key missing in server response."
            else:
                try:
                    # Attempt to parse JSON for detailed error message
                    error_data = response.json()
                    if isinstance(error_data, dict) and "detail" in error_data:
                        return f"Error from server: {error_data['detail']}"
                    # Fallback if "detail" key is not found or JSON is not a dict
                    return f"Error: Server returned status {response.status_code} - {response.text}"
                except JSONDecodeError:  # Changed to use imported JSONDecodeError
                    # Fall through to the generic error message if JSON parsing fails
                    pass
                # Fallback if JSON parsing fails or "detail" is not in a dict
                return f"Error: Server returned status {response.status_code} - {response.text}"
        except requests.exceptions.Timeout:
            return f"Error: Request to {server_url} timed out."
        except requests.exceptions.ConnectionError:
            return f"Error: Failed to connect to {server_url}."
        except Exception as e:
            return f"Error: An unexpected error occurred: {str(e)}."


def get_clients(model: str, session_id: str or None, pandaid: str or None, taskid: str or None, cache: str) -> dict:
    """
    Create and return a dictionary of clients for different categories.

    Args:
        model (str): The model to use for generating answers.
        session_id (str or None): The session ID for the context memory.
        pandaid (str or None): The PanDA ID for the job or task, if applicable.
        taskid (str or None): The task ID for the job or task, if applicable.
        cache (str): The location of the cache directory.

    Returns:
        dict: A dictionary mapping client categories to their respective client classes.
    """
    # PanDA MCP configuration (can be overridden via environment variables)
    panda_mcp_host = os.getenv("PANDA_MCP_HOST", "pandaserver01.sdcc.bnl.gov")
    panda_mcp_port = int(os.getenv("PANDA_MCP_PORT", "25443"))
    panda_mcp_transport = os.getenv("PANDA_MCP_TRANSPORT", "streamable-http")
    panda_mcp_use_http = os.getenv("PANDA_MCP_USE_HTTP", "false").lower() == "true"
    panda_mcp_token = os.getenv("PANDA_MCP_TOKEN")
    panda_mcp_vo = os.getenv("PANDA_MCP_VO", "atlas")

    return {
        "document": DocumentQuery(model, session_id),
        "queue": None,
        "task": TaskStatus(model, taskid, cache, session_id) if session_id and taskid else None,
        "log_analyzer": LogAnalysis(model, pandaid, cache, session_id) if pandaid else None,
        "pilot_activity": None,
        "panda_mcp": PanDAMCPClient(
            host=panda_mcp_host,
            port=panda_mcp_port,
            transport=panda_mcp_transport,
            use_http=panda_mcp_use_http,
            token=panda_mcp_token,
            vo=panda_mcp_vo,
        ),
        "pilot_activity": None,
        "CRIC_analyzer": CRICanalysisClient(schema_path, session_id) # hardcoded schema_path currently
    }


def extract_job_id(text: str) -> int or None:
    """
    Extract a job ID from the given text using a regular expression.

    Args:
        text: The text from which to extract the job ID.

    Returns:
        int or None: The extracted job ID as an integer, or None if no job ID is found.
    """
    pattern = r'\b(?:job|panda[\s_]?id)\s+(\d+)\b'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    return None


def extract_task_id(text: str) -> int or None:
    """
    Extract a task ID from the given text using a regular expression.

    Args:
        text: The text from which to extract the task ID.

    Returns:
        int or None: The extracted task ID as an integer, or None if no task ID is found.
    """
    pattern = r'\b(?:task[\s_]?id|task)\s+(\d+)\b'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    return None


def figure_out_clients(question: str, model: str, session_id: str = None, cache: str = "cache", task_id: int = None, panda_id: int = None) -> dict:
    """
    Determine the appropriate client to handle the given question.

    This function is used by the UI's pipe function to figure out which clients to initialize.

    Args:
        question (str): The question to be answered.
        model (str): The model to use for generating the answer.
        session_id (str): The session ID for the context memory.
        cache (str): The location of the cache directory.
        task_id (str): The task ID, if known.
        panda_id (str): The PanDA ID for the job, if known.
    Returns:
        dict: A dictionary mapping client categories to their respective client classes.
    """
    # does the question contain a job or task id?
    # use a regex to extract "job NNNNN" from args.question
    pandaid = extract_job_id(question) if not panda_id else panda_id
    taskid = extract_task_id(question) if not task_id else task_id
    return get_clients(model, session_id, pandaid, taskid, cache)


def get_id(input_str: str) -> int or None:
    """
    Extract an integer from the user-inputted string, accounting for commas, periods, and spaces.

    Parameters:
        input_str (str): The string containing the integer.

    Returns:
        Optional[int]: The extracted integer, or None if no valid integer is found.
    """

    # Use a regular expression to match numbers with commas, spaces, or periods as separators
    match = re.findall(r'[-+]?\d[\d,.\s]*', input_str)

    # If a match is found, clean the string and convert to int
    if match:
        # Clean the number string by removing commas, spaces, and periods
        cleaned_number = match[0]
        cleaned_number = cleaned_number.replace(',', '').replace(' ', '')
        if cleaned_number.count('.') <= 1:
            stringlist = cleaned_number.split(".")
            intstring = ""
            for number in stringlist:
                intstring += number
            cleaned_number = intstring
        try:
            return int(cleaned_number)
        except ValueError:
            pass  # return None if conversion fails

    # Return None if no match is found or conversion fails
    return None


def main() -> None:
    """
    Parse command-line arguments, call the RAG server, and print the response.

    This function serves as the main entry point for the command-line client.
    It expects two arguments: the question to ask and the model to use.
    It calls the `ask` function to get a response from the RAG server.
    If the `ask` function returns an error (a string prefixed with "Error:"),
    this error is printed to `sys.stderr` and the script exits with status 1.
    Otherwise, the successful answer is printed to `sys.stdout`.

    Raises:
        SystemExit: If the number of command-line arguments is incorrect, or
                    if an error occurs during the RAG server request.
    """
    # Check server health before proceeding
    ec = check_server_health()
    if ec == EC_TIMEOUT:
        logger.warning(f"Timeout while trying to connect to {ASK_PANDA_BASE_URL}.")
        sleep(10)  # Wait for a while before retrying
        ec = check_server_health()
        if ec:
            logger.error("MCP server is not healthy after retry. Exiting.")
            sys.exit(1)
    elif ec:
        logger.error("MCP server is not healthy. Exiting.")
        sys.exit(1)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process some arguments.")

    parser.add_argument('--session-id', type=str, default="None",
                        help='Session ID for the context memory')
    parser.add_argument('--question', type=str, required=True,
                        help='The question to ask the RAG server')
    parser.add_argument('--model', type=str, required=True,
                        help='The model to use for generating the answer')
    parser.add_argument('--cache', type=str, default="cache",
                        help='Location of cache directory (default: cache)')

    args = parser.parse_args()

    # does the question contain a job or task id?
    # use a regex to extract "job NNNNN" from args.question
    pandaid = extract_job_id(args.question)
    if pandaid is not None:
        logger.info(f"Extracted PanDA ID: {pandaid}")
    else:
        logger.info("No PanDA ID found in the question.")
    taskid = extract_task_id(args.question)
    if taskid is not None:
        logger.info(f"Extracted Task ID: {taskid}")
    else:
        logger.info("No Task ID found in the question.")

    clients = get_clients(args.model, args.session_id, pandaid, taskid, args.cache)
    selection_client = Selection(clients, args.model)

    last_question = args.question

    prompt = ""
    if args.session_id != "None":
        # Retrieve context
        history = memory.get_history(args.session_id)
        for user_msg, client_msg in history:
            prompt += f"User: {user_msg}\nAssistant: {client_msg}\n"
    prompt += f"User: {last_question}"
    category = selection_client.answer(last_question)
    client = clients.get(category)

    print("\n", "#"*10)
    logger.info(" CATEGORY \n"+f"{category}")
    logger.info(f"Full question:\n{prompt}")
    print("\n", "#"*10)

    # TODO: Need another agent to summarize the last answer and previous answer, with history memory

    if category == "document":
        logger.info(f"Selected client category: {category} (DocumentQuery)")
        answer = client.ask(args.question)
        logger.info(f"Final answer (document):\n{answer}")
        return answer
    elif category == "log_analyzer":
        logger.info(f"Selected client category: {category} (LogAnalysis)")
        if pandaid is None:
            pandaid = get_id(prompt)
        if pandaid is None:
            err = "Sorry, I need a Job ID to answer questions about jobs."
            logger.info(err)
            return err
        # reinitialize the client with the correct job id
        if not client:
            client = LogAnalysis(args.model, pandaid, args.cache, args.session_id)
        question = client.generate_question("pilotlog.txt")
        answer = client.ask(question)
        logger.info(f"Final answer (log analyzer):\n{answer}")
        return answer
    elif category == "panda_mcp":
        logger.info(f"Selected client category: {category} (PanDAMCPClient)")
        # PanDAMCPClient only needs the latest user question, not the full prompt history.
        answer = client.answer_prompt_sync(args.question)
        logger.info(f"Final answer (panda_mcp):\n{answer}")
        return answer
    elif category == "task":
        logger.info(f"Selected client category: {category} (TaskStatus)")
        if taskid is None:
            taskid = get_id(prompt)
        if taskid is None:
            err = "Sorry, I need a Task ID to answer questions about task status."
            logger.info(err)
            return err
        logger.info(f"Using Task ID: {taskid}")
        # reinitialize the client with the correct task id
        if not client:
            client = TaskStatus(args.model, taskid, args.cache, args.session_id)

        question = client.generate_question()
        answer = client.ask(question)
        logger.info(f"Final answer (task):\n{answer}")
        return answer
    elif category == "CRIC_analyzer":
        logger.info(f"Selected client category: {category} (CRICAnalysis)")
        answer = client.ask(args.question)
        logger.info(f"Final answer (CRIC analyzer):\n{answer}")
        return answer
    else:
        logger.warning("Not yet implemented")
    if client is None:
        return "Sorry, I don’t have enough information to answer that kind of question."

    return "Sorry, I found no client to answer that kind of question."


if __name__ == "__main__":
    main()
