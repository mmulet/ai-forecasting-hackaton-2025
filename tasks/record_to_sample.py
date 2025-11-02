# pyright: standard

from inspect_ai.dataset import  Sample
# from inspect_ai.model import get_model
# in our real test we will use agents, but just use generate for now


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