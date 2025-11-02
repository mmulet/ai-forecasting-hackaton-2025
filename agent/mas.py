# pyright: standard
from pathlib import Path
from agents import Agent as OpenAIAgent, ModelSettings, OpenAIChatCompletionsModel, Runner, RunConfig
from openai import AsyncOpenAI
from inspect_ai.agent import Agent, AgentState, agent, agent_bridge
from inspect_ai.model import messages_to_openai_responses, ChatMessage
import asyncio
from typing import List
import os

from .schema import MultiAgentSystemConfig, AgentConfig
# os.environ["OPENAI_API_KEY"] = "inspectai"
# os.environ["OPENAI_BASE_URL"] = "http://localhost:8000"


# os.environ["OPENAI_API_KEY"] = "dummy-key" #TODO: Find better solution than dummy key
# TODO: Get rid off Incorrect OpenAI API key error message

class MultiAgentSystem:
    """Container and manager for a multi-agent system."""

    # def  __init__(self, config_list: List, model_tag: str, head_agent: Optional[str]):
    def  __init__(self,model: str, config:MultiAgentSystemConfig):
        config_list = config.agents
        model_tag = model
        head_agent = config.head_agent
        self.config_list = config.agents
        self.model = OpenAIChatCompletionsModel(
            model=model_tag,
            openai_client=AsyncOpenAI(
                # base_url="http://localhost:8000/v1",
                base_url="http://localhost:8000",
                # api_key=API_KEY,
                # api_key="token-abc123"
                api_key="inspectai"
            )
        )
        self.agents_dict = self.create_agents(config_list)
        self.head_agent = self.agents_dict[head_agent] if head_agent else self.agents_dict[config_list[0].name]

    def create_agents(self, config_list: List[AgentConfig]):
        agents_dict = {}
        for config in config_list:
            agent = OpenAIAgent(
                name=config.name,
                model=self.model,
                instructions=config.instructions,
                # tools=config.tools,
                handoffs=[agents_dict[agent_name] for agent_name in config.handoffs],
                model_settings=ModelSettings(tool_choice="none")
            )
            # TODO: insert error message if handoff recipient is referenced before instantiation
            agents_dict[config.name] = agent
        #         # TODO: insert error message if handoff recipient is referenced before instantiation
        #     )
        #     agents_dict[config.name] = agent
        return agents_dict

    async def answer_prompt(self, prompt: str, logging=False):
        # TODO: add more logging about the process and individual steps for debugging
        if logging:
            print(f"Answering prompt: {prompt}")
        result = await Runner.run(self.head_agent, prompt)
        if logging:
            print(f"Answer: {result.final_output}")
        return result.final_output
    async def answer_prompt_inspect(self, messages: list[ChatMessage], logging=False) -> AgentState:
        if logging:
            print(f"Answering prompt: {messages}")
        result = await Runner.run(
            starting_agent=self.head_agent,
            input=await messages_to_openai_responses(messages),
            run_config=RunConfig(model="inspect", tracing_disabled=True),
        )
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

    mas = MultiAgentSystem(model_tag, MultiAgentSystemConfig(
        agents=[AgentConfig(**cfg) for cfg in config_list],
        head_agent=head_agent,
    ))
    result = asyncio.run(mas.answer_prompt("Hola, ¿cómo estás?"))
    print(result)

def load_mas(model: str, yaml_config_path: str | Path):
    return MultiAgentSystem(model, MultiAgentSystemConfig.from_yaml(yaml_config_path))


# This is where we define the agent for Inspect AI
@agent
def mas_agent(model: str, yaml: str | Path) -> Agent:
    async def execute(state: AgentState) -> AgentState:
        # Use bridge to map OpenAI Responses API to Inspect Model API
        async with agent_bridge(state) as bridge:
            mas = load_mas(model, yaml)
            await mas.answer_prompt_inspect(state.messages)
           
            return bridge.state

    return execute