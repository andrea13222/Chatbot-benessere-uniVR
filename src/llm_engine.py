from pathlib import Path
from llama_cpp import Llama

BASE_DIR = Path(__file__).resolve().parent.parent

#MODEL_PATH = BASE_DIR / "models" / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
MODEL_PATH = BASE_DIR / "models" / "Qwen2.5-3B-Instruct-Q4_K_M.gguf"
#MODEL_PATH = BASE_DIR / "models" / "phi-4-mini-instruct-q4_k_m.gguf"

_llm = None


#
# def get_llm():
#     global _llm
#     if _llm is None:
#         print(f"[llm_engine] Caricamento modello in memoria...")
#         _llm = Llama(
#             model_path=str(MODEL_PATH),
#             n_ctx=3072,
#             n_threads=3,
#             n_batch=512,
#             verbose=False,
#             n_gpu_layers=-1
#         )
#     return _llm

def get_llm():
    global _llm
    if _llm is None:
        print(f"[llm_engine] Caricamento modello in memoria...")
        _llm = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=2048,
            n_gpu_layers=0,
            n_threads=2,
            n_batch=256,
            verbose=False,

        )
    return _llm
