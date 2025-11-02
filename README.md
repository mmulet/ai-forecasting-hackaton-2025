# ai-forecasting-hackaton-2025
Shared code for The AI Forecasting Hackathon

inspect eval.py contains code to pass to the Inspect eval framework.

Run with

in a separate terminal, then

```bash
uv sync
source .venv/bin/activate
export OPENAI_BASE_URL="http://localhost:8000"
export OPENAI_API_KEY="inspectai"
export OPENAI_TRACING=disabled
inspect eval inspect_eval.py --max-connections 64 --model vllm/unsloth/Qwen2.5-7b-Instruct -M --enable-auto-tool-choice -M --tool-call-parser=openai-json
```
(technically you can start the vllm server from inspect but this has subtle bugs that I really don't want to go into right now)