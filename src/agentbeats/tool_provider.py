import asyncio
import logging
import os

from agentbeats.client import send_message

logger = logging.getLogger(__name__)

MAX_RETRIES = int(os.getenv("A2A_MAX_RETRIES", "1"))
RETRY_BACKOFF_SEC = float(os.getenv("A2A_RETRY_BACKOFF_SEC", "1.0"))
RETRY_BACKOFF_MULT = float(os.getenv("A2A_RETRY_BACKOFF_MULT", "2.0"))


class ToolProvider:
    def __init__(self):
        self._context_ids = {}

    async def talk_to_agent(self, message: str, url: str, new_conversation: bool = False, files: list[str] = None) -> str:
        """
        Communicate with another agent by sending a message and receiving their response.

        Args:
            message: The message to send to the agent
            url: The agent's URL endpoint
            new_conversation: If True, start fresh conversation; if False, continue existing conversation

        Returns:
            str: The agent's response message
        """
        context_id = None if new_conversation else self._context_ids.get(url, None)
        attempts = 0

        while True:
            outputs = await send_message(message=message, base_url=url, context_id=context_id, files=files)
            if outputs.get("status", "completed") == "completed":
                self._context_ids[url] = outputs.get("context_id", None)
                return outputs["response"]

            attempts += 1
            error_text = str(outputs)
            logger.warning("Agent call failed (attempt %s/%s): %s", attempts, MAX_RETRIES + 1, error_text)

            if "INVALID_ARGUMENT" in error_text or "input token count" in error_text:
                raise RuntimeError(f"{url} responded with: {outputs}")

            if attempts > MAX_RETRIES:
                raise RuntimeError(f"{url} responded with: {outputs}")

            await asyncio.sleep(RETRY_BACKOFF_SEC * (RETRY_BACKOFF_MULT ** (attempts - 1)))

    def reset(self):
        self._context_ids = {}
