import os, sys, time, torch
sys.path.insert(0, "/CosyVoice")
sys.path.insert(0, "/CosyVoice/third_party/Matcha-TTS")

def main():
    # Export model
    from cosyvoice.cli.cosyvoice import AutoModel
    MODEL_DIR = "/models/CosyVoice3"
    EXPORT_DIR = "/tmp/cosyvoice3_vllm"

    if not os.path.exists(os.path.join(EXPORT_DIR, "model.safetensors")):
        model = AutoModel(model_dir=MODEL_DIR, load_trt=False, load_vllm=False, fp16=False)
        from cosyvoice.utils.file_utils import export_cosyvoice2_vllm
        export_cosyvoice2_vllm(model.model.llm, EXPORT_DIR, model.model.device)
        import shutil
        for f in os.listdir(os.path.join(MODEL_DIR, "CosyVoice-BlankEN")):
            src = os.path.join(MODEL_DIR, "CosyVoice-BlankEN", f)
            dst = os.path.join(EXPORT_DIR, f)
            if os.path.isfile(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
        print(f"Exported", flush=True)

    from vllm import ModelRegistry
    from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
    ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)
    from vllm import LLM, SamplingParams

    print("Loading vLLM LLM on NPU...", flush=True)
    t0 = time.time()
    llm = LLM(
        model=EXPORT_DIR,
        skip_tokenizer_init=True,
        enable_prompt_embeds=True,
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=0.3,
        max_model_len=4096,
        max_num_seqs=1,
    )
    print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

    for S in [100, 500, 1000]:
        prompt_data = {"prompt_embeds": torch.randn(1, S, 896)}
        sp = SamplingParams(temperature=0.8, top_k=20, max_tokens=25)
        t0 = time.perf_counter()
        outputs = llm.generate(prompt_data, sp)
        torch.npu.synchronize()
        elapsed = time.perf_counter() - t0
        tokens = len(outputs[0].outputs[0].token_ids)
        print(f"  S={S}: {tokens} tok in {elapsed:.2f}s ({elapsed/tokens*1000:.0f}ms/tok)", flush=True)

if __name__ == '__main__':
    main()
