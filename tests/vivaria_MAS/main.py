#!/usr/bin/env python3
"""
Multi-Agent System (MAS) for Vivaria

Entry point for running a multi-agent system in Vivaria.
Based on the structure of modular-public but adapted for multi-agent orchestration.
"""

import os
from pathlib import Path

from pyhooks import Hooks
from agents import Agent as OpenAIAgent, ModelSettings, OpenAIChatCompletionsModel, Runner
from openai import AsyncOpenAI
from schema import MultiAgentSystemConfig

hooks = Hooks()


async def main():
    """Main entry point for the MAS Vivaria agent."""
    
    # Get task from Vivaria
    task = await hooks.get_task_instructions()
    
    # Determine which MAS configuration to use
    config_name = os.environ.get("MAS_CONFIG", "single")
    config_path = Path(__file__).parent / "mas_configs" / f"{config_name}.yaml"
    
    if not config_path.exists():
        config_path = Path(__file__).parent / "mas_configs" / "single.yaml"
        await hooks.log(f"Config {config_name} not found, using single.yaml")
    
    # Load configuration
    config = MultiAgentSystemConfig.from_yaml(config_path)
    await hooks.log(f"Loaded config: {config_name} with {len(config.agents)} agents")
    
    # Get model from environment (Vivaria provides this)
    model_name = os.environ.get("MODEL", "gpt-4")
    
    # Create OpenAI client using Vivaria's environment variables
    openai_client = AsyncOpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL", os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")),
        api_key=os.environ.get("OPENAI_API_KEY", ""),
    )
    
    # Create the model
    model = OpenAIChatCompletionsModel(
        model=model_name,
        openai_client=openai_client
    )
    
    # Build agents from config
    agents_dict = {}
    for agent_config in config.agents:
        agent = OpenAIAgent(
            name=agent_config.name,
            model=model,
            instructions=agent_config.instructions,
            handoffs=[agents_dict[h] for h in agent_config.handoffs],
            model_settings=ModelSettings(tool_choice="none")
        )
        agents_dict[agent_config.name] = agent
    
    # Determine head agent
    if config.head_agent:
        head_agent = agents_dict[config.head_agent]
    else:
        head_agent = agents_dict[config.agents[0].name]
    
    await hooks.log(f"Starting MAS with head agent: {head_agent.name}")
    
    # Run the multi-agent system
    result = await Runner.run(
        starting_agent=head_agent,
        input=task,
    )
    
    # Submit the final output to Vivaria
    await hooks.submit(result.final_output)


if __name__ == "__main__":
    hooks.main(main)
