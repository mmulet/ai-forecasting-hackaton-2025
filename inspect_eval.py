# pyright: standard
import os
import re
os.environ["OPENAI_API_KEY"] = "token-abc123"
os.environ["OPENAI_BASE_URL"] = "http://localhost:8000"
from pathlib import Path
from inspect_ai import Task, task
from inspect_ai.dataset import  hf_dataset, Sample
# from inspect_ai.model import get_model
from inspect_ai.scorer import Score, Target, scorer, accuracy
# in our real test we will use agents, but just use generate for now
from inspect_ai.solver import TaskState
from agent import mas_agent

def record_to_sample(record):
    if record["answer_format"] == "MC":
        return Sample(
            input=record["prompt"],
            target=record["groundtruth"],
            # Don't included choices here because it's already in the prompt
            # choices=find_choices(record["prompt"]),
            metadata=dict(
                answer_format="MC",
                answer=record["groundtruth"]
            )
        )
    return Sample(
        input=record["prompt"],
        target=record["groundtruth"],
        metadata=dict(
            answer_format="Open",
            answer=record["groundtruth"]

        )
    )

answer_regex = re.compile(r"Thus, the correct answer is:\s*([A-Z])")
@scorer(metrics=[accuracy()])
def score_sample():
    async def score(state: TaskState, target: Target) -> Score:
        answer = state.metadata["answer"]
        if state.metadata["answer_format"] != "MC":
            raise ValueError("Only MC supported in this scorer")
        mc_answer = answer[:1] # should be A. or B. etc
        answer_regex_match = answer_regex.search(state.output.completion)
        predicted_answer = answer_regex_match.group(1) if answer_regex_match else None
        value = "C" if predicted_answer == mc_answer else "I"
        return Score(value=value,
                        answer=predicted_answer,
                        )

        # if answer_regex_match:
        #     predicted_answer = answer_regex_match.group(1)
        #     if predicted_answer == mc_answer:
        #         return Score(value="C",
        #                         answer=predicted_answer
        #                         )
        #     else:
        #         return Score(value="I",)
        # return Score(value="I",)
        #     # return Score(value="C" if mc_answer in state.output.completion else "I",)
        # raise ValueError("Only MC supported in this scorer")
        #         # TODO use a different model for grading
        # grader_model = get_model()
        # prompt = f"""You are an evaluation AI. Given the question prompt, the ground truth answer, and the model's answer, determine if the model's answer is correct. Respond with "Yes" for correct and "No" for incorrect.

        # Question: {state.input}
        # Ground Truth: {state.target}
        # Model's Answer: {answer}

        # Is the model's answer correct?"""
        # response = await grader_model.generate(prompt)
        # response.completion.lower().strip()
        # if "yes" in response.completion.lower():
        #     return Score(value="C")
        # else:
        #     return Score(value="I")
    return score

@task
def evaluate_model_on_adele() -> Task:
    ds = hf_dataset(
            path="CFI-Kinds-of-Intelligence/ADeLe_battery_v1dot0",
            split="train",
            sample_fields=record_to_sample,
        )
    # get current folder
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    
    # limit Multiple Choice for faster testing
    ds = ds.filter(lambda x: x.metadata["answer_format"] == "MC")  
    small_subset = ds[0:100]
    
    return Task(
        dataset=small_subset,
        # solver=mas_agent(current_dir / "mas_configs" / "single.yaml"),
        solver=mas_agent(current_dir / "mas_configs" / "autocrat_3.yaml"),
        scorer=[score_sample()],
        metadata=dict(config="autocrat_3")

    )

@task
def evaluate_model_on_adele_2() -> Task:
    ds = hf_dataset(
            path="CFI-Kinds-of-Intelligence/ADeLe_battery_v1dot0",
            split="train",
            sample_fields=record_to_sample,
        )
    # get current folder
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    
    # limit Multiple Choice for faster testing
    ds = ds.filter(lambda x: x.metadata["answer_format"] == "MC")  
    small_subset = ds[0:100]
    
    return Task(
        dataset=small_subset = ds[0:100]
,
        # solver=mas_agent(current_dir / "mas_configs" / "single.yaml"),
        solver=mas_agent(current_dir / "mas_configs" / "single.yaml"),
        scorer=[score_sample()],
        metadata=dict(config="single")

    )

@task
def evaluate_model_on_adele_3() -> Task:
    ds = hf_dataset(
            path="CFI-Kinds-of-Intelligence/ADeLe_battery_v1dot0",
            split="train",
            sample_fields=record_to_sample,
        )
    # get current folder
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    
    # limit Multiple Choice for faster testing
    ds = ds.filter(lambda x: x.metadata["answer_format"] == "MC")  
    small_subset = ds[0:100]
    
    return Task(
        dataset=small_subset,
        # solver=mas_agent(current_dir / "mas_configs" / "single.yaml"),
        solver=mas_agent(current_dir / "mas_configs" / "verifier.yaml"),
        scorer=[score_sample()],
        metadata=dict(config="verifier")

    )
  