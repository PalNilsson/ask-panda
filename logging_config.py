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

"""Logging configuration for Ask Panda services."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys


class ServiceFilter(logging.Filter):
    """
    Ensures every record has a 'service' attribute.
    Useful to distinguish server vs clients in one log file.
    """
    def __init__(self, service_name: str):
        super().__init__()
        self.service_name = service_name

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "service"):
            record.service = self.service_name
        return True


def setup_logging(service_name: str = "ask_panda") -> None:
    """
    Configure root logger for the *current process*.

    Call this from each entrypoint:
    - ask_panda_server.py  -> service_name="server"
    - document_query_client.py -> service_name="doc_client"
    - etc.
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "ask_panda.log"

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(service)s] [%(name)s] %(message)s"
    )

    file_handler = RotatingFileHandler(
        log_file, maxBytes=10_000_000, backupCount=5
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(ServiceFilter(service_name))

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(ServiceFilter(service_name))

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Avoid duplicate handlers if called twice
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(stream_handler)
