"""
AskPanDA API Pipe for Open WebUI

This pipe calls the ask-panda HTTP API instead of importing Python code directly.
Much simpler and avoids dependency conflicts!
"""

from pydantic import BaseModel, Field
import requests


class Pipe:
    class Valves(BaseModel):
        ask_panda_url: str = Field(
            default="http://localhost:8000/agent_ask",
            description="Ask PanDA endpoint",
        )
        model: str = Field(
            default="gemini",
            description="LLM model",
        )

    def __init__(self):
        self.name = "Ask PanDA"
        self.valves = self.Valves()

    async def pipe(
        self,
        body: dict,
        __metadata__: dict = None,
        __event_call__: dict = None
    ) -> str:
        # get session id for history memory
        session_id = (__metadata__ or {}).get("chat_id")

        # filter out UI generated follow up questions
        meta_type = (__metadata__ or {}).get("type")
        is_followup = (__event_call__ in {"follow_ups", "followups"}) or (meta_type and meta_type != "user_response")

        if (is_followup):
            print("NO FOLLOW-UPS")
            return {"follow_ups": []}
        # User lastest question
        lastest_question = body["messages"][-1]["content"]


        try:
            r = requests.post(
                self.valves.ask_panda_url,
                json={"question": lastest_question, "model": self.valves.model},
                timeout=90,
            )
            r.raise_for_status()
            return r.json().get("answer", "No answer")
        except Exception as e:
            return f"Error: {e}"

