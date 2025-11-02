import asyncio
import subprocess
import signal
import sys
from openai import AsyncOpenAI
from agents import Agent, CodeInterpreterTool, Runner, trace, OpenAIResponsesModel, OpenAIChatCompletionsModel
from agents.run_context import RunContextWrapper

import os
os.environ["OPENAI_API_KEY"] = "dummy-key-for-tracing"

API_KEY = "hackathon2025"
vllm_process = None

def signal_handler(sig, frame):
    print("\nInterrupted! Stopping vLLM...")
    if vllm_process:
        vllm_process.terminate()
        vllm_process.wait()
    sys.exit(0)

async def main():
    global vllm_process
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Starting vLLM server from second terminal right now, uncomment for automatic setup

    # # Start vLLM server
    # print("### Starting vLLM server...\n")
    # vllm_process = subprocess.Popen([
    #     "vllm", "serve", "unsloth/Qwen2.5-0.5B-Instruct", 
    #     "--api-key", API_KEY
    # ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # # Wait for server to be ready
    # print("Waiting for server to start...")
    # await asyncio.sleep(20)
    
    try:
        # Create custom model with vLLM endpoint
        vllm_model = OpenAIChatCompletionsModel(
            model="unsloth/Qwen2.5-7B-Instruct",
            openai_client=AsyncOpenAI(
                base_url="http://localhost:8000/v1",
                #api_key=API_KEY,
            ),
        )

        spanish_agent = Agent(
            name="Spanish agent",
            model=vllm_model,
            instructions="You only speak Spanish. You must start every message by saying '<Spanish agent>', then continue with your normal message.",
            tools = [],
        )

        english_agent = Agent(
            name="English agent",
            model=vllm_model,
            instructions="You only speak English. You must start every message by saying <English agent>', then continue with your normal message.",
            tools = [],
        )

        triage_agent = Agent(
            name="Triage agent",
            model=vllm_model,
            instructions="Handoff to the appropriate agent based on the language of the request. You must start every message by saying who you handed off to.",
            tools = [],
            handoffs=[spanish_agent, english_agent],
        )

        print("Answering prompt...")
        result = await Runner.run(triage_agent, "Hola, ¿cómo estás?")
        # result = await Runner.run(triage_agent, "Hey, how are you?")
        print(f"Final output: {result.final_output}")


    
    finally:
        # If automatic server setup, then uncomment

        # print("\n### Stopping vLLM server...\n")
        # if vllm_process:
        #     vllm_process.terminate()
        #     try:
        #         vllm_process.wait(timeout=5)
        #     except subprocess.TimeoutExpired:
        #         print("Force killing vLLM...")
        #         vllm_process.kill()
        #         vllm_process.wait()
        # print("Cleanup complete")
        pass


if __name__ == "__main__":
    asyncio.run(main())