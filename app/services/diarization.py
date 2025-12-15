import os
import sys
import torchaudio
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
asr_system_root = os.path.dirname(project_root)
speaker_lib_path = os.path.join(asr_system_root, '3D-Speaker')

if speaker_lib_path not in sys.path:
    sys.path.append(speaker_lib_path)
    print(f"[System] 3D-Speaker path added: {speaker_lib_path}")

try:
    from speakerlab.bin.infer_diarization import Diarization3Dspeaker
except Exception as e:
    print(f"[Critical] 3D-Speaker import failed: {e}")
    Diarization3Dspeaker = None


class DiarizationService:
    def __init__(self):
        self.use_3d = Diarization3Dspeaker is not None
        if self.use_3d:
            print("[AI] Loading 3D-Speaker model...")
            self.diarizer = Diarization3Dspeaker(device="cuda" if torch.cuda.is_available() else "cpu")
        else:
            print("[AI] 3D-Speaker unavailable, using fallback diarization")
            self.diarizer = None

    def separate(self, audio_path, out_dir):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        results = []
        wav, sr = torchaudio.load(audio_path)
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        if self.use_3d:
            try:
                segments = self.diarizer(audio_path)
            except Exception as e:
                print(f"[Error] Diarization inference failed: {e}")
                segments = []
            for i, seg in enumerate(segments):
                start, end, spk_id = seg
                start_frame = int(start * sr)
                end_frame = int(end * sr)
                if end_frame > wav.shape[1]:
                    end_frame = wav.shape[1]
                if end_frame <= start_frame:
                    continue
                sub_wav = wav[:, start_frame:end_frame]
                filename = f"{base_name}_seg{i}_spk{spk_id}.wav"
                save_path = os.path.join(out_dir, filename)
                torchaudio.save(save_path, sub_wav, sr)
                rel_path = os.path.relpath(save_path, os.path.join(project_root, 'app')).replace("\\", "/")
                results.append({
                    "spk": f"spk{spk_id}",
                    "start": float(start),
                    "end": float(end),
                    "file": rel_path
                })
        else:
            total = wav.shape[1]
            dur = float(total) / float(sr) if sr else 0.0
            max_chunk = 15.0
            i = 0
            start = 0.0
            while start < dur:
                end = min(start + max_chunk, dur)
                st_idx = int(start * sr)
                ed_idx = int(end * sr)
                sub_wav = wav[:, st_idx:ed_idx]
                filename = f"{base_name}_seg{i}_spk0.wav"
                save_path = os.path.join(out_dir, filename)
                torchaudio.save(save_path, sub_wav, sr)
                rel_path = os.path.relpath(save_path, os.path.join(project_root, 'app')).replace("\\", "/")
                results.append({
                    "spk": "spk0",
                    "start": float(start),
                    "end": float(end),
                    "file": rel_path
                })
                if end >= dur:
                    break
                start = end
                i += 1
        return results
