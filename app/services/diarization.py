import os
import sys
import torchaudio
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
# 3D-Speaker is located inside my_voice_project
speaker_lib_path = os.path.join(project_root, '3D-Speaker')
speaker_bin_path = os.path.join(speaker_lib_path, 'speakerlab', 'bin')

if speaker_lib_path not in sys.path:
    sys.path.insert(0, speaker_lib_path)
    print(f"[System] 3D-Speaker lib path added: {speaker_lib_path}")

if speaker_bin_path not in sys.path:
    sys.path.insert(0, speaker_bin_path)
    print(f"[System] 3D-Speaker bin path added: {speaker_bin_path}")

try:
    # Explicitly import speakerlab to ensure it's available
    import speakerlab
    # Since bin is in sys.path, we can import infer_diarization directly
    import infer_diarization
    Diarization3Dspeaker = infer_diarization.Diarization3Dspeaker
except ImportError as e:
    print(f"[Critical] 3D-Speaker import failed: {e}")
    # Try to provide more context on failure
    print(f"[Critical] sys.path: {sys.path}")
    Diarization3Dspeaker = None


class DiarizationService:
    def __init__(self):
        self.use_3d = Diarization3Dspeaker is not None
        if self.use_3d:
            print("[AI] Loading 3D-Speaker model...", file=sys.stderr)
            self.diarizer = Diarization3Dspeaker(device="cuda" if torch.cuda.is_available() else "cpu")
        else:
            print("[AI] 3D-Speaker unavailable, using fallback diarization", file=sys.stderr)
            self.diarizer = None

    def separate(self, audio_path, out_dir):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        results = []
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
                
                results.append({
                    "spk": f"spk{spk_id}",
                    "start": float(start),
                    "end": float(end),
                    "file": rel_path
                })
        else:
            print("[AI] Error: 3D-Speaker unavailable. Fallback disabled as per requirement.", file=sys.stderr)
            # Strict requirement: No fallback.
            return []
                
        return results
