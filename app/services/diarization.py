import os
import sys
import torchaudio
import torch
import importlib.util
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

# 添加3D-Speaker到Python路径
speaker_lib_path = os.path.join(project_root, '3D-Speaker')
speaker_bin_path = os.path.join(speaker_lib_path, 'speakerlab', 'bin')

# 确保路径按正确的优先级添加
sys.path.insert(0, speaker_lib_path)
sys.path.insert(0, speaker_bin_path)

print(f"[System] 3D-Speaker lib path added: {speaker_lib_path}", file=sys.stderr)
print(f"[System] 3D-Speaker bin path added: {speaker_bin_path}", file=sys.stderr)

# 尝试导入3D-Speaker模块
Diarization3Dspeaker = None
try:
    # 先尝试直接导入
    from infer_diarization import Diarization3Dspeaker
    print("[System] Successfully imported Diarization3Dspeaker from infer_diarization", file=sys.stderr)
except ImportError as e:
    print(f"[Critical] Failed to import Diarization3Dspeaker: {e}", file=sys.stderr)
    try:
        # 如果直接导入失败，尝试动态导入
        infer_diarization_path = os.path.join(speaker_bin_path, 'infer_diarization.py')
        if os.path.exists(infer_diarization_path):
            spec = importlib.util.spec_from_file_location("infer_diarization", infer_diarization_path)
            infer_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(infer_module)
            Diarization3Dspeaker = infer_module.Diarization3Dspeaker
            print(f"[System] Successfully loaded Diarization3Dspeaker via dynamic import from {infer_diarization_path}", file=sys.stderr)
        else:
            print(f"[Critical] infer_diarization.py not found at {infer_diarization_path}", file=sys.stderr)
    except Exception as e2:
        print(f"[Critical] Dynamic import also failed: {e2}", file=sys.stderr)
except Exception as e_general:
    print(f"[Critical] General error importing Diarization3Dspeaker: {e_general}", file=sys.stderr)


class DiarizationService:
    def __init__(self):
        self.use_3d = Diarization3Dspeaker is not None
        if self.use_3d:
            print("[AI] Loading 3D-Speaker model...", file=sys.stderr)
            try:
                self.diarizer = Diarization3Dspeaker(device="cuda" if torch.cuda.is_available() else "cpu")
                print("[AI] 3D-Speaker model loaded successfully", file=sys.stderr)
            except Exception as e:
                print(f"[AI] Failed to initialize 3D-Speaker: {e}", file=sys.stderr)
                self.use_3d = False
                self.diarizer = None
        else:
            print("[AI] 3D-Speaker unavailable, using fallback diarization", file=sys.stderr)
            self.diarizer = None

    def separate(self, audio_path, out_dir):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        results = []
        spk_embs = {}
        wav, sr = torchaudio.load(audio_path)
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        
        if self.use_3d:
            print("[AI] Using 3D-Speaker for segmentation...", file=sys.stderr)
            try:
                # 3D-Speaker infer_diarization returns list of [start, end, spk_id]
                segments = self.diarizer(audio_path)
                print(f"[AI] 3D-Speaker found {len(segments)} segments.", file=sys.stderr)
            except Exception as e:
                print(f"[Error] Diarization inference failed: {e}", file=sys.stderr)
                segments = []
            
            if not segments:
                print("[AI] Warning: No segments found by 3D-Speaker. Returning empty.", file=sys.stderr)
            
            for i, seg in enumerate(segments):
                start, end, spk_id = seg
                start_frame = int(start * sr)
                end_frame = int(end * sr)
                
                # Boundary checks
                if end_frame > wav.shape[1]:
                    end_frame = wav.shape[1]
                if end_frame <= start_frame:
                    continue
                    
                sub_wav = wav[:, start_frame:end_frame]
                
                # Unified filename: 0000.wav, 0001.wav
                filename = f"{i:04d}.wav"
                save_path = os.path.join(out_dir, filename)
                torchaudio.save(save_path, sub_wav, sr)
                
                # Return path relative to project root
                # project_root is defined at module level
                rel_path = os.path.relpath(save_path, project_root).replace("\\", "/")
                
                spk_str = f"spk{spk_id}"
                results.append({
                    "spk": spk_str,
                    "start": float(start),
                    "end": float(end),
                    "file": rel_path
                })
                
                # Extract embedding
                try:
                    feat = self.diarizer.feature_extractor(sub_wav).unsqueeze(0).to(self.diarizer.device)
                    with torch.no_grad():
                        emb = self.diarizer.embedding_model(feat).detach().squeeze(0).cpu().numpy()
                    if spk_str not in spk_embs:
                        spk_embs[spk_str] = []
                    spk_embs[spk_str].append(emb)
                except Exception as e:
                    print(f"[AI] Embedding extraction failed for segment {i}: {str(e)}", file=sys.stderr)

            # Calculate average normalized embedding per speaker
            spk_emb_avg = {}
            for spk, embs in spk_embs.items():
                avg = np.mean(embs, axis=0)
                avg = avg / (np.linalg.norm(avg) + 1e-6)
                spk_emb_avg[spk] = avg.tolist()
                
            return results, spk_emb_avg
        else:
            print("[AI] Error: 3D-Speaker unavailable. Fallback disabled as per requirement.", file=sys.stderr)
            # Strict requirement: No fallback.
            return [], {}
        
        return results, {}
