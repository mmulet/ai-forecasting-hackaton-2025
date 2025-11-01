import asyncio
import subprocess
import signal
import sys
from openai import AsyncOpenAI
from agents import Agent, CodeInterpreterTool, Runner, trace, OpenAIResponsesModel, OpenAIChatCompletionsModel, function_tool

from tools import run_python

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
        
        agent = Agent(
            name="Coder",
            model=vllm_model,
            instructions="You like coding and have access to a tool to run code.",
            tools=[run_python],
        )

        print("Fetching news...")
        result = await Runner.run(agent, "Produce a Python script that prints the phrase 'Hello World!'.")
        async for event in result.stream_events():
            if (
                event.type == "run_item_stream_event"
                and event.item.type == "tool_call_item"
                and event.item.raw_item.type == "code_interpreter_call"
            ):
                print(f"Code interpreter code:\n```\n{event.item.raw_item.code}\n```\n")
            elif event.type == "run_item_stream_event":
                print(f"Other event: {event.item.type}")

        # Access directly after streaming completes
        print(f"Final output: {result.final_output}")
        
        print(f"Final output: {result.final_output}")


    
    finally:
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