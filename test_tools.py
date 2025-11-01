import asyncio
import subprocess
import signal
import sys
from openai import AsyncOpenAI
from agents import Agent, CodeInterpreterTool, Runner, trace, OpenAIResponsesModel, OpenAIChatCompletionsModel, function_tool
from duckduckgo_search import DDGS

import os
os.environ["OPENAI_API_KEY"] = "dummy-key-for-tracing"

API_KEY = "hackathon2025"
vllm_process = None

@function_tool
def get_news_articles(topic):
    print(f"Running DuckDuckGo news search for {topic}...")
    
    # DuckDuckGo search
    ddg_api = DDGS()
    results = ddg_api.text(f"{topic} ", max_results=5)
    if results:
        news_results = "\n\n".join([f"Title: {result['title']}\nURL: {result['href']}\nDescription: {result['body']}" for result in results])
        print(news_results)
        return news_results
    else:
        return f"Could not find news results for {topic}."

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
            name="Newsie",
            model=vllm_model,
            instructions="You have access to the internet to search for news.",
            tools=[get_news_articles],
        )

        print("Fetching news...")
        #result = await Runner.run(agent, "Produce a Python script that prints the phrase 'Hello World!'.")
        result = Runner.run_streamed(agent, "What is currently going on in the world? Answer in five sentences.'.")
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