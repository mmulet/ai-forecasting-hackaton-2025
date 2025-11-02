# pyright: standard

from .record_to_sample import record_to_sample
from inspect_ai.dataset import  hf_dataset

def load_dataset():
    ds = hf_dataset(
            path="CFI-Kinds-of-Intelligence/ADeLe_battery_v1dot0",
            split="train",
            sample_fields=record_to_sample,
        )
    return ds.filter(lambda x: x.metadata["answer_format"] == "MC")  