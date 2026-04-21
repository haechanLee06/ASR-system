import time
import sys
import os
import json

# Add project root to sys.path
base_dir = os.path.dirname(os.path.abspath(__file__))
# Note: we will be running this inside WSL, so we need to adjust path
sys.path.append(base_dir)

def test_speed():
    print("[TEST] Initializing AIServiceRunner...")
    start_init = time.time()
    from app.services.ai_service import AIServiceRunner
    runner = AIServiceRunner()
    # Mock some data to avoid full diarization
    runner.init_models()
    end_init = time.time()
    print(f"[TEST] Initialization took: {end_init - start_init:.2f}s")

    # Test file
    test_file = "app/static/separated/107/0001.wav"
    abs_test_file = os.path.join(base_dir, test_file)
    
    if not os.path.exists(abs_test_file):
        print(f"[ERROR] Test file not found: {abs_test_file}")
        return

    print(f"[TEST] Transcribing {test_file}...")
    start_asr = time.time()
    # Simulation of only FunASR part
    proc_path = runner._preprocess_audio(abs_test_file)
    res_b = runner.pipeline_b.generate(input=proc_path)
    text = res_b[0].get('text', '')
    
    # Optional: Punctuation
    if runner.punc_model:
        res_p = runner.punc_model.generate(input=text)
        text = res_p[0].get('text', text)
        
    end_asr = time.time()
    print(f"[TEST] Transcription took: {end_asr - start_asr:.2f}s")
    print(f"[RESULT] Text: {text}")

if __name__ == "__main__":
    test_speed()
