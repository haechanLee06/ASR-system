
import os
import sys
import torch
import torchaudio
import traceback
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# Set up simple logging
import logging
logging.basicConfig(level=logging.INFO)

def run_test_diarization(audio_path):
    print(f"[Test] Starting Diarization Test for: {audio_path}")
    
    if not os.path.exists(audio_path):
        print(f"[Error] File not found: {audio_path}")
        return

    print("[Test] Loading Pipeline...")
    try:
        diar_pipeline = pipeline(
            task=Tasks.speaker_diarization,
            model='damo/speech_campplus_speaker-diarization_common'
        )
    except Exception as e:
        print(f"[Error] Failed to load pipeline: {e}")
        traceback.print_exc()
        return

    # Test 1: Default Parameters
    print("\n[Test 1] Running with DEFAULT parameters...")
    try:
        res1 = diar_pipeline(audio_path)
        print(f"Result 1 Raw: {res1}")
    except Exception as e:
        print(f"[Error] Test 1 failed: {e}")

    # Test 2: Aggressive VAD
    print("\n[Test 2] Running with AGGRESSIVE VAD (min_on=0.1, min_off=0.1)...")
    try:
        res2 = diar_pipeline(audio_path, min_on=0.1, min_off=0.1)
        print(f"Result 2 Raw: {res2}")
    except Exception as e:
        print(f"[Error] Test 2 failed: {e}")

    # Test 3: Aggressive VAD + Max Duration Limit
    print("\n[Test 3] Running with VAD + Max Dur=10.0...")
    try:
        # Note: 'max_dur' might not be a direct arg for this specific pipeline, but we test if it accepts it or kwargs
        res3 = diar_pipeline(audio_path, min_on=0.1, min_off=0.1, max_dur=10.0)
        print(f"Result 3 Raw: {res3}")
    except Exception as e:
        print(f"[Error] Test 3 failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_diarization_only.py <audio_path>")
    else:
        run_test_diarization(sys.argv[1])
