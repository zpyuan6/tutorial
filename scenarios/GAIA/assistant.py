import argparse
import uvicorn
from dotenv import load_dotenv
import logging
import os

load_dotenv()

from google.adk.agents import Agent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from tools import write_file, visit_webpage, read_file_from_artifact, read_pdf,read_text_file,read_excel,inspect_image, get_stock_prices,execute_python,get_wikipedia_history
from typing import List
import hashlib

from google.genai.types import Part
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse

from a2a.types import (
    AgentCapabilities,
    AgentCard,
)

SUPPORTED_INLINE_MIMES = {
    # documents (Gemini doc support is PDF + text) :contentReference[oaicite:2]{index=2}
    "application/pdf",
    "text/plain",
    # images (Gemini supports png/jpeg/webp) :contentReference[oaicite:3]{index=3}
    "image/png",
    "image/jpeg",
    "image/webp",
}

async def before_model_callback(callback_context, llm_request):
    for content in llm_request.contents:
        modified_parts: List[Part] = []

        print("Before process: ", content.parts)
        print(callback_context)

        for part in content.parts:
            # 1) Tetx part：save directly
            if getattr(part, "text", None):
                modified_parts.append(part)
                continue

            # 2) Uploaded files：saved under part.inline_data
            if getattr(part, "inline_data", None):
                processed = await _process_inline_data_part(part, callback_context)
                modified_parts.extend(processed)
            else:
                # 3) Other parts (e.g., function_response) are not processed here, just keep them as is
                modified_parts.append(part)

        content.parts = modified_parts

    print("After process", content.parts)
    print(callback_context)

    return None

async def _process_inline_data_part(
    part: Part,
    callback_context: CallbackContext,
) -> List[Part]:
    """Convert inline_data into artifact, inject marker + keep original part."""
    artifact_id = _generate_artifact_id(part)

    # If artifact does not exist, save it (official codelab also lists then saves):contentReference[oaicite:5]{index=5}
    if artifact_id not in await callback_context.list_artifacts():
        await callback_context.save_artifact(filename=artifact_id, artifact=part)

    marker = Part(
        text=(
            f"[User Uploaded Artifact] filename={artifact_id}, mime={part.inline_data.mime_type}. "
            f"Use tool read_file_from_artifact to load this artifact by filename."
        )
    )

    return [marker]

def _generate_artifact_id(part: Part) -> str:
    inline = part.inline_data
    mime = getattr(inline, "mime_type")
    data = inline.data or b""
    h = hashlib.sha1()
    h.update(mime.encode("utf-8"))
    h.update(b"\n")
    h.update(data)
    return f"usr_upl_{h.hexdigest()}"

def main():
    parser = argparse.ArgumentParser(description="Run the A2A debater agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9019, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL to provide in the agent card")
    args = parser.parse_args()
    my_tools = [write_file, visit_webpage, read_file_from_artifact, get_stock_prices,execute_python, get_wikipedia_history]

    assistant_model = os.getenv("ASSISTANT_MODEL", "gemini-2.0-flash")
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting GAIA assistant with model=%s", assistant_model)
    root_agent = Agent(
        name="assistant",
        model=assistant_model,
        description="Response user queries.",
        # instruction="You are a professional assistant.",
        instruction=(
            "You are a capable Green Agent for the GAIA benchmark.\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. **NEVER GIVE UP:** Do not say 'I cannot do this'. Use your tools to find a way.\n"
            "2. **WIKIPEDIA HISTORY:** If asked about **edit history**, **revisions**, or **who edited a page**, "
            "you MUST use `execute_python` to fetch data using the `wikipedia` library or `requests` (Wikimedia API).\n"
            "   - Example logic: `import wikipedia; page = wikipedia.page('Michael Myers (Halloween)'); print(page.url)`\n"
            "   - Then use `requests` to query the API: `https://en.wikipedia.org/w/api.php?action=query&prop=revisions...`\n"
            # "3. **FILES:** Check file extensions. Use `read_text_file`, `read_excel`, `read_pdf`, or `inspect_image` accordingly.\n"
            "3. **FILES:** If user provide files, such as excel, PDF, txt, and images. Use `read_file_from_artifact` to understand the content.\n"
            "4. **PYTHON:** Use `execute_python` for math, data analysis, and scraping.\n"
            "5. **LIMIT OUTPUT:** When using `execute_python`, print only final values or small slices; never print full page content.\n"
            "6. **NO GUESTS:** Do not hallucinate content. Read the source."
        ),
        # You are an autonomous AI agent, consisting of a large language model and a scaffolding program that runs on a linux VM.
        # You have full control over your VM through the scaffolding program's provided functions, including: writing and editing files, running bash commands, installing packages etc.. From your VM, you can access the internet.
        # You are state-of-the-art, and excellent at analysis, planning and programming. You possess encyclopedic knowledge of the world. You write in a very direct and concise style.
        # You are resourceful and adaptive, and you never give up. Before deciding something can't be done, you try it out. You consider multiple options and choose the best one. If your current approach doesn't work, you formulate a new plan. You are given a task you need to solve completely on your own.
        # Please think step by step before calling tools. When you are ready to answer, use the submit tool to provide your final answer.
        tools=my_tools,
        before_model_callback = before_model_callback,
    )

    agent_card = AgentCard(
        name="assistant",
        description='Response user queries.',
        url=args.card_url or f'http://{args.host}:{args.port}/',
        version='1.0.0',
        default_input_modes=[
            'text',
            'file'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        # skills=['file_reading', 'file_writing'],
        skills=[],
    )

    a2a_app = to_a2a(root_agent, agent_card=agent_card)
    uvicorn.run(a2a_app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
