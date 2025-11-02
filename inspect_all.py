import os
# os.environ["OPENAI_API_KEY"] = "inspectai"
# os.environ["OPENAI_BASE_URL"] = "http://localhost:8000"

from inspect_ai import eval_async
from tasks import single, autocrat, verifier


async def main():
    model = "unsloth/Qwen2.5-14B-Instruct"
    vllm_model = f"vllm/{model}"
    coroutines = [
        eval_async(single(model),model=vllm_model, max_connections=32, model_args={"device": "2"}),
        # eval_async(autocrat(model),model=vllm_model, max_connections=64, model_args={"device": "3"}),
        # eval_async(verifier(model),model=vllm_model, max_connections=64, model_args={"device": "4"})
    ]
    await asyncio.gather(*coroutines)
    print("done!")

    


    # await eval_async(single(),model="vllm/unsloth/Qwen2.5-7B-Instruct", max_connections=64, model_args={"device": "2"})
    # print("done!")

    # coros = [
    #     eval_async(single()),
    #     eval_async(autocrat()),
    #     eval_async(verifier())
    # ]
    # results = await asyncio.gather(*coros)
    # results["single"] = results[0]
    # results["autocrat"] = results[1]
    # results["verifier"] = results[2]

    # for task_name, result in results.items():
    #     print(f"Results for task: {task_name}")
    #     print(result.metrics)
    #     print("-" * 40)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())