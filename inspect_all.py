import asyncio
import multiprocessing as mp
from typing import Any, Callable, Dict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED, Future
from collections import deque

from inspect_ai import eval_async
from tasks import single, autocrat, verifier

TASKS: dict[str, Callable[[str], Any]] = {
    "single": single,
    "autocrat": autocrat,
    "verifier": verifier,
}

def run_eval(task_name: str, model: str, gpu: int, max_connections: int = 32) -> int:
    vllm_model = f"vllm/{model}"
    task_fn = TASKS[task_name]

    async def _run() -> None:
        await eval_async(
            task_fn(model),
            model=vllm_model,
            max_connections=max_connections,
            model_args={"device": str(gpu)},
        )

    asyncio.run(_run())
    return gpu  # hand GPU back to the scheduler


class QueuedJobs:
    def __init__(self, configs: Any, devices: list[int]):
        self.configs = configs
        self.devices = devices
        self.parallelism = len(devices)

        self.job_queue = deque(configs)
        self.free_devices = deque(devices)
        self.in_flight: Dict[Future[int], int] = {}
    def add_jobs(self, ex: ProcessPoolExecutor, ):
        while self.job_queue and self.free_devices and len(self.in_flight) < self.parallelism:
            task_name, mdl, max_conns = self.job_queue.popleft()
            gpu = self.free_devices.popleft()
            fut = ex.submit(run_eval, task_name, mdl, gpu, max_conns)
            self.in_flight[fut] = gpu


def main() -> None:
    # List all jobs (task_name, model, max_connections)
    configs = [        
        ("single", "unsloth/Qwen2.5-3B-Instruct", 128),
        ("autocrat", "unsloth/Qwen2.5-3B-Instruct", 128),
        ("verifier", "unsloth/Qwen2.5-3B-Instruct", 128),

        ("single", "mistralai/Mistral-7B-Instruct-v0.2", 128),
        ("autocrat", "mistralai/Mistral-7B-Instruct-v0.2", 128),
        ("verifier", "mistralai/Mistral-7B-Instruct-v0.2", 128),
    ]
    devices = [0, 1, 2, 3, 4, 5, 6, 7]  # GPUs you want to use



    q = QueuedJobs(configs, devices)

 
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(devices), mp_context=ctx) as ex:
        # Prime
        q.add_jobs(ex)

        # Feed as GPUs free up
        while q.in_flight or q.job_queue:
            done, _ = wait(q.in_flight.keys(), return_when=FIRST_COMPLETED)
            for fut in done:
                gpu = q.in_flight.pop(fut)
                try:
                    returned_gpu = fut.result()
                except Exception as e:
                    print(f"[GPU {gpu}] failed: {e}")
                    returned_gpu = gpu
                q.free_devices.append(returned_gpu)

                q.add_jobs(ex)

    print("done!")

if __name__ == "__main__":
    main()