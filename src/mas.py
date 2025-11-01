from agents import Agent, OpenAIChatCompletionsModel, Runner
from openai import AsyncOpenAI

import asyncio

from typing import List, Dict, Set, Optional, Callable

import os
os.environ["OPENAI_API_KEY"] = "dummy-key" #TODO: Find better solution than dummy key
# TODO: Get rid off Incorrect OpenAI API key error message

class MultiAgentSystem:
    """Container and manager for a multi-agent system."""

    def  __init__(self, config_list: List, model_tag: str, head_agent: Optional[str]):
        self.config_list = config_list
        self.model = OpenAIChatCompletionsModel(
            model=model_tag,
            openai_client=AsyncOpenAI(
                base_url="http://localhost:8000/v1",
                #api_key=API_KEY,
            )
        )
        self.agents_dict = self.create_agents(config_list)
        self.head_agent = self.agents_dict[head_agent] if head_agent else self.agents_dict[config_list[0]["name"]]

    def create_agents(self, config_list):
        agents_dict = {}
        for config in config_list:
            agent = Agent(
                name = config.get("name"),
                model = self.model,
                instructions = config.get("instructions"),
                tools = config.get("tools", []),
                handoffs = [agents_dict[agent_name] for agent_name in config.get("handoffs", [])]
                # TODO: insert error message if handoff recipient is referenced before instantiation
            )
            agents_dict[config.get("name")] = agent
        return agents_dict

    async def answer_prompt(self, prompt: str, logging=False):
        # TODO: add more logging about the process and individual steps for debugging
        if logging:
            print(f"Answering prompt: {prompt}")
        result = await Runner.run(self.head_agent, prompt)
        if logging:
            print(f"Answer: {result.final_output}")
        return result.final_output
    
if __name__ == "__main__":
    config_list = [
        {
            "name": "Spanish agent",
            "instructions":"You only speak Spanish. You must start every message by saying '<Spanish agent>', then continue with your normal message.",
        },
        {
            "name": "English agent",
            "instructions":"You only speak English. You must start every message by saying <English agent>', then continue with your normal message."
        },
        {
            "name": "Triage agent",
            "instructions": "Handoff to the appropriate agent based on the language of the request. You must start every message by saying who you handed off to.",
            "handoffs": ["Spanish agent", "English agent"]
        },
    ]
    head_agent = "Triage agent"
    model_tag = "unsloth/Qwen2.5-7B-Instruct"

    mas = MultiAgentSystem(config_list, model_tag, head_agent)
    result = asyncio.run(mas.answer_prompt("Hola, ¿cómo estás?"))
    print(result)

def load_mas(yaml):
    with open(yaml, 'r') as f:
        config = yaml.saf_load(f)
        config_list = config.get('agents')
        model_tag = config.get('model_tag')
        head_agent = config.get('head_agent', None)

        mas = MultiAgentSystem(config_list, model_tag, head_agent)
        return mas