import sys
import os

# Insert modeling folder to sys.path so vLLM can import it
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "Qwen3.5-0.8B")))

# Dynamically register custom architecture in vLLM registry
try:
    from vllm.model_executor.models.registry import ModelRegistry
    ModelRegistry.register_model("Qwen2ForCausalLMWithTriton", "modeling_qwen:Qwen2ForCausalLMWithTriton")
    print("[INFO] Registered Qwen2ForCausalLMWithTriton. Custom kernels remain opt-in via USE_CUSTOM_RMSNORM/USE_CUSTOM_SWIGLU.")
except Exception as e:
    print(f"[WARNING] Failed to register custom architecture in vLLM registry: {e}")

from vllm.entrypoints.cli.main import main
if __name__ == "__main__":
    main()
