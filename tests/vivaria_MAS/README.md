# Multi-Agent System (MAS) for Vivaria

This folder contains a Vivaria agent that runs a multi-agent system. It follows the same structure as modular-public but implements multi-agent orchestration using the `agents` library.

## Structure

```
vivaria/
├── main.py              # Entry point (uses pyhooks)
├── manifest.json        # Agent metadata
├── requirements.txt     # Dependencies
├── schema.py            # Configuration schema
└── mas_configs/         # Agent configurations
    ├── single.yaml
    ├── autocrat_3.yaml
    └── verifier.yaml
```

## Usage

### Run in Vivaria

From the repository root:

```bash
viv run <task> --agent-path vivaria/
```

### Use a specific configuration

```bash
viv run <task> --agent-path vivaria/ --env MAS_CONFIG=autocrat_3
```

### Available configurations

- `single` - Single agent (default)
- `autocrat_3` - 4-agent autocratic system  
- `verifier` - 2-agent verification system (Guesser + Verifier)

## How It Works

1. `main.py` is the entry point (called via `hooks.main(main)`)
2. Loads a YAML config from `mas_configs/` based on `MAS_CONFIG` env var
3. Creates multiple agents with handoff capabilities
4. Runs the head agent on the task
5. Agents can hand off to each other as needed
6. Submits final output via `hooks.submit()`

## Environment Variables

- `MAS_CONFIG` - Which configuration to use (default: `single`)
- `MODEL` - Which model to use (default: `gpt-4`)
- `OPENAI_API_KEY` - Set by Vivaria automatically
- `OPENAI_BASE_URL` - Set by Vivaria automatically

## Differences from modular-public

- **modular-public**: Single agent with tools
- **This agent**: Multiple agents that can hand off to each other

The MAS approach enables:
- Specialized agents for different subtasks
- Agent collaboration and verification
- Complex multi-step problem-solving workflows
