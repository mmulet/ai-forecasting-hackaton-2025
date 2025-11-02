from datasets import Dataset

def make_seed() -> int:
    """Create a random seed based on the current time."""
    return int(time.time() * 1000) % 2**32