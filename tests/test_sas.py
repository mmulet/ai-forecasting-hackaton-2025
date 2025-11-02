import asyncio
import subprocess
import signal
import sys
from openai import AsyncOpenAI
from agents import Agent, CodeInterpreterTool, Runner, trace, OpenAIResponsesModel, OpenAIChatCompletionsModel

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
            name="Code interpreter",
            model=vllm_model,
            instructions="You love doing math.",
            # tools=[
            #     CodeInterpreterTool(
            #         tool_config={"type": "code_interpreter", "container": {"type": "auto"}},
            #     )
            # ],
            tools = [],
        )

        print("Solving math problem...")
        result = await Runner.run(agent, "What is the square root of 273 * 312821 plus 1782?")
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