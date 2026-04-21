import time
import sys
import os

# Add project root to sys.path
base_dir = "/mnt/c/Users/asus/Desktop/graduate-project/ASR-system/my_voice_project"
sys.path.append(base_dir)

def test_funasr_only():
    from funasr import AutoModel
    import torch
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    models_dir = os.path.join(base_dir, "models")
    path_b = os.path.join(models_dir, "lukeewin01", "paraformer-large-sichuan-offline")
    path_vad = os.path.join(models_dir, "iic", "speech_fsmn_vad_zh-cn-16k-common-pytorch")
    
    print(f"[TEST] Device: {device}")
    print("[TEST] Loading FunASR Paraformer...")
    s1 = time.time()
    model = AutoModel(
        model=path_b,
        vad_model=path_vad,
        device=device,
        disable_update=True
    )
    s2 = time.time()
    print(f"[TEST] Model Load Time: {s2 - s1:.2f}s")
    
    test_file = os.path.join(base_dir, "app/static/separated/107/0001.wav")
    print(f"[TEST] Transcribing {test_file}...")
    s3 = time.time()
    res = model.generate(input=test_file)
    s4 = time.time()
    print(f"[TEST] Transcription Time: {s4 - s3:.2f}s")
    print(f"[RESULT] Text: {res[0]['text']}")

if __name__ == "__main__":
    test_funasr_only()
