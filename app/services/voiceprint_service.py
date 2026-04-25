import os
import sys
import json
import traceback

def extract_embedding(audio_path):
    import torch
    import torchaudio
    import numpy as np
    
    base_dir_mod = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if base_dir_mod not in sys.path:
        sys.path.append(base_dir_mod)
    
    from app.services.diarization import speaker_lib_path, speaker_bin_path
    if speaker_lib_path not in sys.path:
        sys.path.insert(0, speaker_lib_path)
    if speaker_bin_path not in sys.path:
        sys.path.insert(0, speaker_bin_path)
    
    from infer_diarization import get_speaker_embedding_model
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    embedding_model, feature_extractor = get_speaker_embedding_model(device)
    
    wav, sr = torchaudio.load(audio_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        wav = resampler(wav)
    
    # Check energy, if silent return errors
    energy = float(np.mean(np.abs(wav.numpy()))) if wav.numel() > 0 else 0.0
    if energy < 0.001:
        raise ValueError("音频过轻或为空")
        
    feat = feature_extractor(wav).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = embedding_model(feat).detach().squeeze(0).cpu().numpy()
        
    emb = emb / (np.linalg.norm(emb) + 1e-6)
    return emb.tolist()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: python voiceprint_service.py <audio_path>"}))
        sys.exit(1)
        
    try:
        import platform
        audio_path = sys.argv[1]
        if platform.system().lower().startswith("win"):
            from app.utils.wsl_bridge import run_in_wsl
            res = run_in_wsl("app/services/voiceprint_service.py", audio_path)
            print(json.dumps(res, ensure_ascii=False))
            sys.exit(0)
            
        emb = extract_embedding(audio_path)
        print(json.dumps({"embedding": emb}, ensure_ascii=False))
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
