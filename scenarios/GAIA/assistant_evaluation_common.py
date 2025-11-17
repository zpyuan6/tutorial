from pydantic import BaseModel
from typing import Literal

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

class ResponseEval(BaseModel):
    final_answer: str
    ground_truth: str
    is_correct: bool
    query_level: str
    time_to_answer_sec: float

class AssistantEval(BaseModel):
    score: float
    average_time_to_answer_sec: float
    responses_records: list[ResponseEval]



def assistant_evaluation_agent_card(agent_name: str, card_url: str) -> AgentCard:
    skill = AgentSkill(
        id='assistant_evaluation',
        name='Seek and evaluate responses to user queries',
        description='Seek and evaluate responses from General AI Assistant to user queries with fundamental abilities such as reasoning, multi-modality handling, web browsing, and generally tool-use proficiency. ',
        tags=['debate'],
        examples=["""
                  {
                    "participants": {
                      "assistant": "https://assistant.example.com:443",
                    },
                    "config": {
                      "query": "A paper about AI regulation that was originally submitted to arXiv.org in June 2022 shows a figure with three axes, where each axis has a label word at both ends. Which of these words is used to describe a type of society in a Physics and Society article submitted to arXiv.org on August 11, 2016?",
                    }
                  }
                  """]
    )
    agent_card = AgentCard(
        name=agent_name,
        description='Seek and evaluate responses from General AI Assistant to user queries with fundamental abilities such as reasoning, multi-modality handling, web browsing, and generally tool-use proficiency.',
        url=card_url,
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )
    return agent_card
