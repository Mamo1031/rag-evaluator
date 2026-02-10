from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

import requests
from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    host: str
    api_key: str
    project_id: int


def load_config() -> Config:
    load_dotenv()
    host = os.getenv("HOST")
    api_key = os.getenv("API_KEY")
    project_id = os.getenv("PROJECT_ID")
    if not host:
        raise RuntimeError("HOST is not set (check your .env file).")
    if not api_key:
        raise RuntimeError("API_KEY is not set (check your .env file).")
    if not project_id:
        raise RuntimeError("PROJECT_ID is not set (check your .env file).")
    if not host.startswith("http"):
        raise RuntimeError("HOST must be a URL starting with https://.")
    if not host.endswith("/"):
        host += "/"
    if not api_key.startswith("Bearer "):
        api_key = f"Bearer {api_key.strip()}"
    try:
        project_id_int = int(project_id)
    except ValueError as exc:
        raise RuntimeError("PROJECT_ID must be an integer.") from exc
    return Config(host=host, api_key=api_key, project_id=project_id_int)


class ChatClient:
    def __init__(self, config: Config):
        self.config = config

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": self.config.api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def chat(
        self,
        user_input: str,
        *,
        model_variant: str = "gpt-4o-mini",
        template: str = "",
        data_source_items: Optional[list[dict]] = None,
        stream: bool = False,
        chat_id: str = "00000000-0000-0000-0000-000000000000",
        timeout: float = 60,
    ) -> requests.Response:
        payload = {
            "action": "new",
            "projectId": self.config.project_id,
            "chat_id": chat_id,
            "data_source_items": data_source_items or [],
            "model_variant": model_variant,
            "template": template,
            "user_input": user_input,
        }
        url = f"{self.config.host}/api/chat/"
        timeout_tuple = (10, timeout)
        response = requests.post(
            url,
            headers=self._headers(),
            json=payload,
            stream=stream,
            timeout=timeout_tuple,
        )
        response.raise_for_status()
        return response

    @staticmethod
    def iter_ndjson_lines(response: requests.Response) -> Iterator[dict]:
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def extract_final_message(events: Iterable[dict]) -> Optional[str]:
    final_message = None
    for event in events:
        if event.get("type") == "replace_message":
            final_message = event.get("message")
    return final_message
