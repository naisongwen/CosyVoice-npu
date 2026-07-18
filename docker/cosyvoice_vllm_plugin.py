"""Auto-register CosyVoice2ForCausalLM with vLLM ModelRegistry.
Runs in both the main process and spawned EngineCore processes."""
def register():
    import sys, os
    root = os.environ.get("COSYVOICE_ROOT", "/workspace/CosyVoice")
    if root not in sys.path:
        sys.path.insert(0, root)
    matcha = os.path.join(root, "third_party", "Matcha-TTS")
    if os.path.isdir(matcha) and matcha not in sys.path:
        sys.path.insert(0, matcha)
    from vllm import ModelRegistry
    from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
    ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)
