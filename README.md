# ai-forecasting-hackaton-2025
Shared code for The AI Forecasting Hackathon

inspect eval.py contains code to pass to the Inspect eval framework.

1. Go to https://huggingface.co/datasets/CFI-Kinds-of-Intelligence/ADeLe_battery_v1dot0 and accept the terms of use.

2. Put your huggingface token in .env file with 
```
HF_TOKEN=your_token_here
```

Run with

in a separate terminal, then
NEW
```bash
uv sync
source .venv/bin/activate
python inspect_all.py
```


--------
OLD

```bash
uv sync
source .venv/bin/activate
export OPENAI_BASE_URL="http://localhost:8000"
export OPENAI_API_KEY="inspectai"
export OPENAI_TRACING=disabled
inspect eval inspect_eval.py --max-connections 64 --model vllm/unsloth/Qwen2.5-7B-Instruct
```
(technically you can start the vllm server from inspect but this has subtle bugs that I really don't want to go into right now)

inspect eval inspect_eval.py --max-connections 64 --model vllm/unsloth/Qwen2.5-7B-Instruct


<!-- inspect eval inspect_eval.py --max-connections 64 --model vllm/unsloth/Qwen2.5-7b-Instruct -M enable-auto-tool-choice -M tool-call-parser=openai-json -->