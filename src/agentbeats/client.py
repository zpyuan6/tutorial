import asyncio
import json
import logging
from uuid import uuid4

import httpx
from a2a.client import (
    A2ACardResolver,
    ClientConfig,
    ClientFactory,
    Consumer,
)
from a2a.types import (
    Message,
    Part,
    Role,
    TextPart,
    DataPart,
    FilePart, 
    FileWithBytes
)
import base64
import mimetypes
from pathlib import Path


DEFAULT_TIMEOUT = 300

def make_file_part(path: str) -> Part:
    p = Path(path)
    mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")

    # FileWithBytes has fields: bytes, name, mime_type :contentReference[oaicite:1]{index=1}
    f = FileWithBytes(bytes=b64, name=p.name, mime_type=mime)
    return Part(root=FilePart(file=f))

def create_message(*, role: Role = Role.user, text: str, context_id: str | None = None, files: list[str] = None) -> Message:

    if files and len(files) > 0:
        parts = [Part(TextPart(kind="text", text=text))]
        parts += [make_file_part(path) for path in files]
        return Message(
            kind="message",
            role=role,
            parts=parts,
            message_id=uuid4().hex,
            context_id=context_id
        )
    else:
        return Message(
            kind="message",
            role=role,
            parts=[Part(TextPart(kind="text", text=text))],
            message_id=uuid4().hex,
            context_id=context_id
        )

def merge_parts(parts: list[Part]) -> str:
    chunks = []
    for part in parts:
        if isinstance(part.root, TextPart):
            chunks.append(part.root.text)
        elif isinstance(part.root, DataPart):
            chunks.append(json.dumps(part.root.data, indent=2))
    return "\n".join(chunks)

async def send_message(message: str, base_url: str, context_id: str | None = None, streaming=False, consumer: Consumer | None = None, files: list[str] = None) -> dict:
    """Returns dict with context_id, response and status (if exists)"""
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
        agent_card = await resolver.get_agent_card()
        config = ClientConfig(
            httpx_client=httpx_client,
            streaming=streaming,
        )
        factory = ClientFactory(config)
        client = factory.create(agent_card)
        if consumer:
            await client.add_event_consumer(consumer)

        outbound_msg = create_message(text=message, context_id=context_id, files=files)
        last_event = None
        outputs = {
            "response": "",
            "context_id": None
        }

        # if streaming == False, only one event is generated
        async for event in client.send_message(outbound_msg):
            last_event = event

        match last_event:
            case Message() as msg:
                outputs["context_id"] = msg.context_id
                outputs["response"] += merge_parts(msg.parts)

            case (task, update):
                outputs["context_id"] = task.context_id
                outputs["status"] = task.status.state.value
                msg = task.status.message
                if msg:
                    outputs["response"] += merge_parts(msg.parts)
                if task.artifacts:
                    for artifact in task.artifacts:
                        outputs["response"] += merge_parts(artifact.parts)

            case _:
                pass

        return outputs
