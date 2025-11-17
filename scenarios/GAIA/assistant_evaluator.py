import argparse 
import contextlib
import uvicorn
import asyncio
import logging
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Literal

load_dotenv()

from google import genai
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    TaskState,
    Part,
    TextPart,
)
from a2a.utils import (
    new_agent_text_message
)

from agentbeats.green_executor import GreenAgent, GreenExecutor
from agentbeats.models import EvalRequest, EvalResult
from agentbeats.tool_provider import ToolProvider
import pandas as pd


from assistant_evaluation_common import ResponseEval, AssistantEval, assistant_evaluation_agent_card


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debate_judge")


class GAIAAssistantEvaluator(GreenAgent):
    def __init__(self):
        self._required_roles = ["assistant"]
        self._required_config_keys = ["evaluation_level"]
        self._client = genai.Client()  # Initialize your AI client here
        self._tool_provider = ToolProvider()

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self._required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"
        missing_config_keys = set(self._required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"
        
        if not request.config["evaluation_level"] in ["all","l1", "l2"]:
            return False, f"Incorrect evaluation level setting, plese select one of following level, 'all', 'l1', or 'l2'."

        return True, "ok"

    async def run_eval(self, req: EvalRequest, updater: TaskUpdater) -> None:
        # Implementation of the evaluation logic goes here
        logger.info(f"Starting GAIA assistant evaluation: {req}")

        try:

            if req.config["evaluation_level"] == "all":
                splits = {
                    'test': '2023/test/metadata.parquet', 
                    'validation': '2023/validation/metadata.parquet'
                    }
            elif req.config["evaluation_level"] == "l1":
                splits = {
                    'test': '2023/test/metadata.level1.parquet', 
                    'validation': '2023/validation/metadata.level1.parquet'
                    } 
            elif req.config["evaluation_level"] == "l2":
                splits = {
                    'test': '2023/test/metadata.level2.parquet', 
                    'validation': '2023/validation/metadata.level2.parquet'
                    }
            
            df = pd.read_parquet(
                "hf://datasets/gaia-benchmark/GAIA/" + splits["test"])
            
            df_iterrows = df.iterrows()
            items_number = 0
            correct_number = 0
            sum_time_consumed = 0
            response_records = []
            
            for index, item in df_iterrows:
                user_query = item['Question']
                query_level = item['Level']
                ground_truth = item['Final answer']
                attached_file = item['file_path']
                
                start_time = asyncio.get_event_loop().time()
                response = await self.assistant_response(
                    req.participants,
                    user_query,
                    attached_file,
                    updater,
                )
                time_consumed = asyncio.get_event_loop().time() - start_time
                sum_time_consumed += time_consumed

                await updater.update_status(
                    TaskState.working, 
                    new_agent_text_message(f"Assistant response: {response}")
                    )
                
                logger.info(f"Assistant response obtained. Evaluating response.")
                isCorrect: bool = await self.evaluate_response(
                    user_query,
                    response['assistant'][0],
                    ground_truth
                )


                if isCorrect:
                    correct_number+=1
                items_number += 1

                response_eval = ResponseEval(
                    final_answer = response['assistant'][0],
                    ground_truth = ground_truth,
                    is_correct = isCorrect,
                    query_level = query_level,
                    time_to_answer_sec = time_consumed,
                )
                response_records.append(response_eval)
                logger.info(f"Response Evaluation:\n{response_eval.model_dump_json()}")
            
            assistant_eval = AssistantEval(
                score = correct_number/items_number,
                average_time_to_answer_sec= sum_time_consumed/items_number,
                responses_records=response_records
            )

            logger.info(f"Assistant Evaluation: \n Query Number: {items_number} \n Score: {correct_number/items_number} \n Average Time: {sum_time_consumed/items_number}  ")

            await updater.add_artifact(
                parts=[
                    Part(root=TextPart(text=assistant_eval.model_dump_json()))
                ],
                name ="Result",
            )

        finally:
            self._tool_provider.reset()


    async def assistant_response(
        self,
        participants: dict[str, str],
        query: str,
        attached_file: str,
        updater: TaskUpdater,
    ) -> dict[str, list[str]]:
        responses: dict[str, list[str]] = {"assistant": []}

        async def turn(role: str, prompt: str) -> str:
            response = await self._tool_provider.talk_to_agent(prompt, str(participants[role]), new_conversation=True)
            logger.info(f"{role}: {response}")
            responses[role].append(response)
            await updater.update_status(TaskState.working, new_agent_text_message(f"{role}: {response}"))
            return response

        # Opening turns
        r = await turn("assistant", f"User query: {query}. Response user query.")

        return  responses
    
    async def evaluate_response(self, user_query: str, response: str, ground_truth: str) -> bool:
        # prompt adapted from GAIA evaluation guidelines:

        system_prompt = """
                        You are an expert evaluator for AI assistants. 
                        Your task is to evaluate the assistant's response to a user query based on the provided ground truth response. 
                        Provide a bool value to indicate correctness for generated responses.
                        """
        
        user_prompt = f"""
                        Evaluate the response from the AI assistant to the user query: '{user_query}'
                        Response: '{response}'
                        Ground Truth: '{ground_truth}'
                        Provide a bool value to indicate correctness for generated responses.
                        """
        
        response = self._client.models.generate_content(
            model = "gemini-2.5-flash",
            config = genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=bool,
            ),
            contents = user_prompt
        )

        return response.parsed


async def main():
    parser = argparse.ArgumentParser(description="Run the A2A debate judge.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9019, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL to provide in the agent card")
    parser.add_argument("--cloudflare-quick-tunnel", action="store_true", help="Use a Cloudflare quick tunnel. Requires cloudflared. This will override --card-url")
    args = parser.parse_args()

    if args.cloudflare_quick_tunnel:
        from agentbeats.cloudflare import quick_tunnel
        agent_url_cm = quick_tunnel(f"http://{args.host}:{args.port}")
    else:
        agent_url_cm = contextlib.nullcontext(args.card_url or f"http://{args.host}:{args.port}/")

    async with agent_url_cm as agent_url:
        agent = GAIAAssistantEvaluator()
        executor = GreenExecutor(agent)
        agent_card = assistant_evaluation_agent_card("AssistantEvaluator", agent_url)

        request_handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=InMemoryTaskStore(),
        )

        server = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler,
        )

        uvicorn_config = uvicorn.Config(server.build(), host=args.host, port=args.port)
        uvicorn_server = uvicorn.Server(uvicorn_config)
        await uvicorn_server.serve()

if __name__ == '__main__':

    # Login using e.g. `huggingface-cli login` to access this dataset
    asyncio.run(main())