from runpod_flash import Endpoint, GpuType
import os

HF_TOKEN = os.getenv("HF_TOKEN")

@Endpoint(
    name="llm-inference",
    gpu=GpuType.NVIDIA_GEFORCE_RTX_4090,
    dependencies=["vllm", "torch"],
    env={"HF_TOKEN": HF_TOKEN}
)
async def llm(prompt: str) -> str:
    from vllm import LLM, SamplingParams

    model = LLM("meta-llama/Llama-3.1-8B-Instruct")
    params = SamplingParams(max_tokens=512, temperature=0.7)
    out = model.generate(prompt, params)
    return out[0].outputs[0].text