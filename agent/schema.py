from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel
import yaml


class AgentConfig(BaseModel):
    name: str
    instructions: str
    # No support for loading dataclasses
    # from yaml., will have to add toolse
    # manually
    # tools: List[str] = []
    handoffs: List[str] = []
    

class MultiAgentSystemConfig(BaseModel):
    agents: List[AgentConfig]
    head_agent: Optional[str] = None
    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "MultiAgentSystemConfig":
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)