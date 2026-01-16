import argparse
import contextlib
import uvicorn
import asyncio
import logging
import math
import os
import random
import re
import io
import json
import requests
from dotenv import load_dotenv
import pandas as pd
from datasets import load_dataset
import editdistance
from PIL import Image
from tools import question_scorer

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.join(CURRENT_FILE_DIR, "workspace")
os.makedirs(WORKSPACE_DIR, exist_ok=True)
RESULTS_DIR = os.path.join(WORKSPACE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

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

from assistant_evaluation_common import ResponseEval, AssistantEval, assistant_evaluation_agent_card

GAIA_DATASET = "gaia-benchmark/GAIA"  # verify
DOCVQA_DATASET = "lmms-lab/DocVQA"  # verify
DOCVQA_CONFIG = "DocVQA"  # verify
DOCVQA_SPLIT = "validation"
SEALQA_DATASET = "vtllms/sealqa"  # verify
SEALQA_SPLIT = "test"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GAIA Benchmarking")


class GAIAAssistantEvaluator(GreenAgent):
    def __init__(self):
        self._required_roles = ["assistant"]
        self._required_config_keys = ["evaluation_level"]
        # api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        # self._client = genai.Client(api_key=api_key)
        self._tool_provider = ToolProvider()
        self._eval_model = os.getenv("EVALUATOR_MODEL", "gemini-2.0-flash")

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self._required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"
        missing_config_keys = set(self._required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"

        if request.config["evaluation_level"] not in ["all", "l1", "l2", "l3"]:
            return False, "Incorrect evaluation level setting, please select one of 'all', 'l1', 'l2', or 'l3'."

        return True, "ok"

    def download_file(self, filename, split="validation"):
        """Downloading the file from huggingface"""
        if not filename:
            return None

        local_filename = os.path.basename(filename)
        local_path = os.path.join(WORKSPACE_DIR, local_filename)
        print("local_filename", local_filename)

        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            return local_path

        base_url = "https://hf-mirror.com/datasets/gaia-benchmark/GAIA/resolve/main/"

        if filename.startswith("2023/"):
            url = base_url + filename
        else:
            url = base_url + f"2023/{split}/{filename}"

        logger.info(f"downloading {filename} ...")

        headers = {}
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"

        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                with open(local_path, "wb") as f:
                    f.write(resp.content)
                logger.info(f"download successfully {local_path}")
                return local_path
            else:
                logger.error(f" download failed Status {resp.status_code}: {url}")
                return None
        except Exception as e:
            logger.error(f" {e}")
            return None

    def _normalize_text(self, text: str) -> str:
        text = (text or "").lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def _anls_score(self, pred: str, golds: list[str]) -> float:
        if not golds:
            return 0.0
        pred_norm = self._normalize_text(pred)
        best = 0.0
        for gold in golds:
            gold_norm = self._normalize_text(gold)
            denom = max(len(pred_norm), len(gold_norm), 1)
            dist = editdistance.eval(pred_norm, gold_norm)
            score = 1.0 - (dist / denom)
            if score < 0.5:
                score = 0.0
            best = max(best, score)
        return best

    def _is_safe_refusal(self, answer: str) -> bool:
        if not answer:
            return False
        text = answer.lower()
        refusal_markers = [
            "i don't know",
            "i do not know",
            "cannot",
            "can't",
            "unable",
            "not sure",
            "unsure",
            "no information",
            "no info",
            "unknown",
        ]
        return any(marker in text for marker in refusal_markers)

    def _save_docvqa_image(self, image_obj, filename: str) -> str | None:
        if image_obj is None:
            return None
        try:
            if hasattr(image_obj, "convert"):
                img = image_obj
            elif isinstance(image_obj, dict) and "bytes" in image_obj:
                img = Image.open(io.BytesIO(image_obj["bytes"]))
            elif isinstance(image_obj, (bytes, bytearray)):
                img = Image.open(io.BytesIO(image_obj))
            else:
                return None

            img = img.convert("RGB")
            local_path = os.path.join(WORKSPACE_DIR, filename)
            img.save(local_path, format="JPEG", quality=95)
            return filename
        except Exception as e:
            logger.error(f"DocVQA image save failed for {filename}: {e}")
            return None

    def _sample_indices(self, total: int, num_samples: int, seed: int) -> list[int]:
        if total <= 0:
            return []
        num_samples = max(0, min(num_samples, total))
        rng = random.Random(seed)
        return rng.sample(range(total), num_samples)

    def _parse_bool(self, value: object, default: bool = True) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in ("1", "true", "yes", "y", "on"):
                return True
            if normalized in ("0", "false", "no", "n", "off"):
                return False
        return default

    async def run_eval(self, req: EvalRequest, updater: TaskUpdater) -> None:
        logger.info(f"Starting Extending GAIA assistant evaluation: {req}")

        try:
            run_gaia = self._parse_bool(req.config.get("run_gaia"), True)
            run_docvqa = self._parse_bool(req.config.get("run_docvqa"), True)
            run_sealqa = self._parse_bool(req.config.get("run_sealqa"), True)

            if run_gaia:
                gaia_result = await self._run_gaia_eval(req, updater)
            else:
                logger.info("Skipping GAIA evaluation (run_gaia=false)")
                gaia_result = {
                    "skipped": True,
                    "evaluation_level": req.config.get("evaluation_level"),
                    "split": req.config.get("gaia_split"),
                    "num_items": 0,
                    "score": {
                        "Level 1": 0.0,
                        "Level 2": 0.0,
                        "Level 3": 0.0,
                        "Total": 0.0,
                    },
                    "average_time_to_answer_sec": 0.0,
                    "responses_records": [],
                    "errors": [],
                }

            if run_docvqa:
                docvqa_result = await self._run_docvqa_eval(req, updater)
            else:
                logger.info("Skipping DocVQA evaluation (run_docvqa=false)")
                docvqa_result = {
                    "skipped": True,
                    "dataset": DOCVQA_DATASET,
                    "split": DOCVQA_SPLIT,
                    "num_samples": 0,
                    "actual_num_samples": 0,
                    "seed": req.config.get("docvqa_seed"),
                    "anls": 0.0,
                    "records": [],
                }

            if run_sealqa:
                sealqa_result = await self._run_sealqa_eval(req, updater)
            else:
                logger.info("Skipping SEALQA evaluation (run_sealqa=false)")
                sealqa_result = {
                    "skipped": True,
                    "dataset": SEALQA_DATASET,
                    "subset": req.config.get("sealqa_subset"),
                    "split": SEALQA_SPLIT,
                    "num_samples": 0,
                    "strict_accuracy": 0.0,
                    "safe_refusal_rate": 0.0,
                    "records": [],
                }

            num_tasks = sum([run_gaia, run_docvqa, run_sealqa])
            capability_score = (
                (gaia_result.get("score").get("Total", 0.0) if run_gaia else 0.0) +
                (sealqa_result.get("strict_accuracy", 0.0) if run_sealqa else 0.0) +
                (docvqa_result.get("anls", 0.0) if run_docvqa else 0.0)
            ) / num_tasks if num_tasks > 0 else 0.0

            # total_token_cost = float(req.config.get("total_token_cost", 0))
            # denom = math.log10(total_token_cost + 1) if total_token_cost > 0 else 1.0
            # green_index = capability_score / denom

            result = {
                "gaia": gaia_result,
                "docvqa": docvqa_result,
                "sealqa": sealqa_result,
                "capability_score": capability_score,
                # "green_index": green_index,
                # "total_token_cost": total_token_cost,
            }

            await updater.add_artifact(
                parts=[
                    Part(root=TextPart(text=json.dumps(result, ensure_ascii=True)))
                ],
                name="Result",
            )
            try:
                output_path = os.path.join(RESULTS_DIR, "result.json")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=True, indent=2)
                logger.info("Wrote result file to %s", output_path)
            except Exception as e:
                logger.error("Failed to write result file: %s", e)

            try:
                errors_output = {
                    "gaia": gaia_result.get("errors", []),
                    "docvqa": [r for r in docvqa_result.get("records", []) if "error" in r],
                    "sealqa": [r for r in sealqa_result.get("records", []) if "error" in r],
                }
                errors_path = os.path.join(RESULTS_DIR, "errors.json")
                with open(errors_path, "w", encoding="utf-8") as f:
                    json.dump(errors_output, f, ensure_ascii=True, indent=2)
                logger.info("Wrote error file to %s", errors_path)
            except Exception as e:
                logger.error("Failed to write error file: %s", e)

        finally:
            self._tool_provider.reset()

    async def _run_gaia_eval(self, req: EvalRequest, updater: TaskUpdater) -> dict:
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Starting GAIA evaluation"),
        )

        if req.config["evaluation_level"] == "all":
            splits = {
                "test": "2023/test/metadata.parquet",
                "validation": "2023/validation/metadata.parquet",
            }
        elif req.config["evaluation_level"] == "l1":
            splits = {
                "test": "2023/test/metadata.level1.parquet",
                "validation": "2023/validation/metadata.level1.parquet",
            }
        elif req.config["evaluation_level"] == "l2":
            splits = {
                "test": "2023/test/metadata.level2.parquet",
                "validation": "2023/validation/metadata.level2.parquet",
            }
        else:
            splits = {
                "test": "2023/test/metadata.level3.parquet",
                "validation": "2023/validation/metadata.level3.parquet",
            }

        gaia_split = str(req.config.get("gaia_split", "validation"))
        if gaia_split not in splits:
            gaia_split = "validation"

        df = pd.read_parquet(
            f"hf://datasets/{GAIA_DATASET}/" + splits[gaia_split]
        )

        items_number = {'1':0, '2':0, '3':0}
        correct_number = {'1':0, '2':0, '3':0}
        sum_time_consumed = 0.0
        response_records = []
        errors = []

        total_df_rows = len(df)

        for index, item in df.iterrows():
            user_query = item.get("Question", "")
            query_level = item.get("Level", "")
            ground_truth = item.get("Final answer", "")
            attached_file = item.get("file_path")
            items_number[query_level] += 1

            logger.info(
                f"GAIA query index={index+1}/{total_df_rows}, level={query_level}, have_attached_file={bool(attached_file)}, question={user_query}"
                )

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"GAIA query index={index+1}/{total_df_rows}, level={query_level}, have_attached_file={bool(attached_file)}, question={user_query}"),
            )
            
            available_file = self.download_file(attached_file, split=gaia_split) if attached_file else None

            start_time = asyncio.get_event_loop().time()
            try:
                response = await self.assistant_response(
                    req.participants,
                    user_query,
                    available_file,
                )
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Assistant response: {response}. Ground truth: {ground_truth}."),
                )
            except Exception as e:
                errors.append({"index": int(index), "error": str(e), "question": user_query})
                logger.error(f"GAIA query index={index} failed: {e.__traceback__}")
                response_records.append(
                    ResponseEval(
                        final_answer="",
                        ground_truth=str(ground_truth),
                        is_correct=False,
                        query_level=str(query_level),
                        time_to_answer_sec=0.0,
                    )
                )
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"GAIA item failed (index={index}): {e}"),
                )

            time_consumed = asyncio.get_event_loop().time() - start_time
            sum_time_consumed += time_consumed

            logger.info(f"Assistant response: {response}. Ground truth: {ground_truth}.")

            is_correct = question_scorer(
                response['assistant'][0],
                ground_truth,
            )

            if is_correct:
                correct_number[query_level] += 1

            response_eval = ResponseEval(
                final_answer=response["assistant"][0],
                ground_truth=str(ground_truth),
                is_correct=is_correct,
                query_level=str(query_level),
                time_to_answer_sec=time_consumed,
            )
            response_records.append(response_eval)
            logger.info(f"Response Evaluation:\n{response_eval.model_dump_json()}")


        total_query_number = sum(items_number.values())
        total_correct = sum(correct_number.values())
        scores = {f"Level {level}": (correct_number[level] / items_number[level]) if items_number[level] > 0 else None for level in items_number.keys()}
        scores["Total"] = (total_correct / total_query_number) if total_query_number > 0 else 0.0
        avg_time = sum_time_consumed / total_query_number

        logger.info(
            "Assistant Evaluation: \n"
            f"Query Number: {items_number} \n"
            f"Score: {scores} \n"
            f"Average Time: {avg_time}"
        )

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"GAIA benchmarking completed. Score: {scores}, Average Time: {avg_time} seconds."),
        )

        return {
            "evaluation_level": req.config.get("evaluation_level"),
            "split": gaia_split,
            "num_items": items_number,
            "score": scores,
            "average_time_to_answer_sec": avg_time,
            "responses_records": [r.model_dump() for r in response_records],
            "errors": errors,
        }

    async def _run_docvqa_eval(self, req: EvalRequest, updater: TaskUpdater) -> dict:
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Starting DocVQA evaluation"),
        )

        num_samples = int(req.config.get("docvqa_num_samples", 200))
        seed = int(req.config.get("docvqa_seed", 0))
        dataset = load_dataset(DOCVQA_DATASET, DOCVQA_CONFIG, split=DOCVQA_SPLIT)
        indices = self._sample_indices(len(dataset), num_samples, seed)

        scores = []
        records = []

        for idx in indices:
            sample = dataset[idx]
            question = sample.get("question") or sample.get("question_text")
            if not question:
                records.append({
                    "id": idx,
                    "error": "missing_question",
                })
                continue

            answers = sample.get("answers") or sample.get("answer")
            if answers is None:
                answers = []
            if isinstance(answers, str):
                answers = [answers]
            else:
                answers = list(answers)

            image_obj = sample.get("image") or sample.get("document") or sample.get("doc")
            filename = f"docvqa_{idx}.jpg"
            saved = self._save_docvqa_image(image_obj, filename)
            if not saved:
                records.append({
                    "id": idx,
                    "question": question,
                    "gold": answers,
                    "error": "image_save_failed",
                })
                continue
            
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"DOCVQA query index={idx+1}/{num_samples}, question={question}"),
            )

            try:
                response = await self.assistant_response(
                    req.participants,
                    question,
                    filename,
                )
                pred = response["assistant"][0] if response["assistant"] else ""

                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Assistant response: {response}. Ground truth: {answers}."),
                )
            except Exception as e:
                records.append({
                    "id": sample.get("question_id") or sample.get("questionId") or idx,
                    "question": question,
                    "gold": answers,
                    "error": str(e),
                    "image_filename": filename,
                })
                continue
            score = self._anls_score(pred, answers)
            scores.append(score)

            records.append({
                "id": sample.get("question_id") or sample.get("questionId") or idx,
                "question": question,
                "pred": pred,
                "gold": answers,
                "anls": score,
                "image_filename": filename,
            })


        anls = (sum(scores) / len(scores)) if scores else 0.0

        return {
            "dataset": DOCVQA_DATASET,
            "split": DOCVQA_SPLIT,
            "num_samples": num_samples,
            "actual_num_samples": len(scores),
            "seed": seed,
            "anls": anls,
            "records": records,
        }

    async def _run_sealqa_eval(self, req: EvalRequest, updater: TaskUpdater) -> dict:
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Starting SEALQA evaluation"),
        )

        subset = str(req.config.get("sealqa_subset", "seal_0"))
        dataset = load_dataset(SEALQA_DATASET, name=subset, split=SEALQA_SPLIT)

        records = []
        correct_count = 0
        refusal_count = 0
        total = 0

        for idx, sample in enumerate(dataset):
            question = sample.get("question") or sample.get("query") or sample.get("prompt")
            answers = sample.get("answer") or sample.get("answers")
            if answers is None:
                answers = []
            if isinstance(answers, str):
                answers = [answers]
            else:
                answers = list(answers)

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"SEALQA query index={idx+1}/{len(dataset)}, question={question}"),
            )

            if not question:
                records.append({
                    "id": idx,
                    "error": "missing_question",
                })
                continue

            try:
                response = await self.assistant_response(
                    req.participants,
                    question,
                    None,
                )
                pred = response["assistant"][0] if response["assistant"] else ""
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Assistant response: {response}. Ground truth: {answers}."),
                )
            except Exception as e:
                total += 1
                records.append({
                    "id": sample.get("id") or idx,
                    "question": question,
                    "gold": answers,
                    "error": str(e),
                })
                continue

            pred_norm = self._normalize_text(pred)
            gold_norms = [self._normalize_text(a) for a in answers]
            is_correct = any(pred_norm == g for g in gold_norms if g)
            is_safe_refusal = self._is_safe_refusal(pred)

            if is_correct:
                correct_count += 1
            if is_safe_refusal:
                refusal_count += 1
            total += 1

            records.append({
                "id": sample.get("id") or idx,
                "question": question,
                "pred": pred,
                "gold": answers,
                "is_correct": is_correct,
                "is_safe_refusal": is_safe_refusal,
            })

        strict_accuracy = (correct_count / total) if total else 0.0
        refusal_rate = (refusal_count / total) if total else 0.0

        return {
            "dataset": SEALQA_DATASET,
            "subset": subset,
            "split": SEALQA_SPLIT,
            "num_samples": total,
            "strict_accuracy": strict_accuracy,
            "safe_refusal_rate": refusal_rate,
            "records": records,
        }

    async def assistant_response(
        self,
        participants: dict[str, str],
        query: str,
        attached_file: str,
    ) -> dict[str, list[str]]:
        responses: dict[str, list[str]] = {"assistant": []}

        async def turn(role: str, prompt: str, files: list[str]=None) -> str:
            response = await self._tool_provider.talk_to_agent(
                    prompt, str(participants[role]), new_conversation=True, files=files 
                )
            logger.info(f"{role}: {response}")
            responses[role].append(response)

            return response

        prompt = f"User query: {query}. \n Response user query."

        if attached_file:
            # prompt += (
            #     "\n\n[System Notice] A file has been downloaded to your local workspace."
            #     f"\nFilename: {attached_file}"
            #     f"\nPath: workspace/{attached_file}"
            #     "\nUse the appropriate tool based on file type: read_text_file, read_excel, read_pdf, or inspect_image."
            # )
            await turn("assistant", prompt, [attached_file])
        else:
            # prompt += " Response user query."
            await turn("assistant", prompt)

        return responses


async def main():
    parser = argparse.ArgumentParser(description="Run the A2A debate judge.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL to provide in the agent card")
    parser.add_argument(
        "--cloudflare-quick-tunnel",
        action="store_true",
        help="Use a Cloudflare quick tunnel. Requires cloudflared. This will override --card-url",
    )
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


if __name__ == "__main__":
    asyncio.run(main())
