import argparse
import uvicorn
from dotenv import load_dotenv
import logging

load_dotenv()

from google.adk.agents import Agent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from tools import write_file, visit_webpage, read_pdf,read_text_file,read_excel,inspect_image, get_stock_prices,execute_python,get_wikipedia_history

from a2a.types import (
    AgentCapabilities,
    AgentCard,
)

def main():
    parser = argparse.ArgumentParser(description="Run the A2A debater agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9019, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL to provide in the agent card")
    args = parser.parse_args()
    my_tools = [write_file, visit_webpage, read_pdf,read_text_file,read_excel,inspect_image, get_stock_prices,execute_python, get_wikipedia_history]

    root_agent = Agent(
        name="assistant",
        model="gemini-2.0-flash",
        description="Response user queries.",
        # instruction="You are a professional assistant.",
        instruction=(
            # "You are a professional assistant capable of handling files. "
            # "When a user mentions a file path or asks a question involving a file, "
            # "IMMEDIATELY use the `read_file` tool to inspect its content before answering. "
            # "Do not guess the file content."
            "You are a capable Green Agent for the GAIA benchmark.\n"
            "TOOLS USAGE PROTOCOL:\n"
            # "1. **FILES:** If the user mentions a file path, use `read_file` immediately.\n"
            # "2. **WEBSITES:** If the user provides a URL (e.g., 'Check https://github.com...'), "
            # "IMMEDIATELY use the `visit_webpage` tool to read its content. Do not ask for permission.\n"
            # "3. **UNKNOWN INFO:** If asked about a specific topic (e.g., 'Wikipedia history of X','real time information','specific books'), "
            # "you should construct the URL yourself to search it(e.g., 'https://en.wikipedia.org/wiki/X') "
            # "and use `visit_webpage` to read it.\n"
            # "1. **FILES:** Use `read_text_file`, `read_excel`, `read_pdf`, `inspect_image` based on extension.\n"
            # "2. **WEBSITES:** Use `visit_webpage`.\n"
            # "3. **STOCK MARKET / FINANCE:** If asked about stock prices, trends, or market data:\n"
            # "   - Identify the Ticker Symbol (e.g., Apple -> AAPL).\n"
            # "   - Use the `get_stock_prices` tool.\n"
            # "   - DO NOT say you cannot access real-time data.\n"
            "You are a highly capable Green Agent for the GAIA benchmark.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. **NEVER GIVE UP:** Do not say 'I cannot do this'. Use your tools to find a way.\n"
            "2. **WIKIPEDIA HISTORY:** If asked about **edit history**, **revisions**, or **who edited a page**, "
            "you MUST use `execute_python` to fetch data using the `wikipedia` library or `requests` (Wikimedia API).\n"
            "   - Example logic: `import wikipedia; page = wikipedia.page('Michael Myers (Halloween)'); print(page.url)`\n"
            "   - Then use `requests` to query the API: `https://en.wikipedia.org/w/api.php?action=query&prop=revisions...`\n"
            "3. **FILES:** Check file extensions. Use `read_text_file`, `read_excel`, `read_pdf`, or `inspect_image` accordingly.\n"
            "4. **PYTHON:** Use `execute_python` for math, data analysis, and scraping."
            "5. **NO GUESTS:** Do not hallucinate content. Read the source."
        ),
        tools=my_tools
    )

    agent_card = AgentCard(
        name="assistant",
        description='Response user queries.',
        url=args.card_url or f'http://{args.host}:{args.port}/',
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        # skills=['file_reading', 'file_writing'],
        skills=[],
    )

    a2a_app = to_a2a(root_agent, agent_card=agent_card)
    uvicorn.run(a2a_app, host=args.host, port=args.port)
    logging.basicConfig(level=logging.DEBUG)


if __name__ == "__main__":
    main()
