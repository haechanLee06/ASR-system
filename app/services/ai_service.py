import sys
import os
import json
import traceback
import re
import logging
import platform

# 屏蔽繁杂的日志
logging.getLogger('modelscope').setLevel(logging.ERROR)
logging.getLogger('funasr').setLevel(logging.ERROR)

def setup_env():
    try:
        from funasr import AutoModel
        import torch
        import torchaudio
        import numpy as np
        return AutoModel, torch, torchaudio, np
    except ImportError as e:
        print(json.dumps({"error": f"Environment Missing: {e}"}))
        sys.exit(1)

class AIServiceRunner:
    def __init__(self):
        self.AutoModel, self.torch, self.torchaudio, self.np = setup_env()
        self.device = "cuda:0" if self.torch.cuda.is_available() else "cpu"
        self.pipeline_b = None
        base_dir_mod = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if base_dir_mod not in sys.path:
            sys.path.append(base_dir_mod)
        from app.services.diarization import DiarizationService
        self.diarizer = DiarizationService()

    def init_models(self):
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            models_dir = os.path.join(base_dir, "models")
            print(f"[AI] base_dir={base_dir}")
            print(f"[AI] models_dir={models_dir}")
            if not os.path.exists(models_dir):
                raise RuntimeError(f"Models folder not found at {models_dir}")
            path_b = os.path.join(models_dir, "lukeewin01", "paraformer-large-sichuan-offline")
            path_vad = os.path.join(models_dir, "iic", "speech_fsmn_vad_zh-cn-16k-common-pytorch")
            path_punc = os.path.join(models_dir, "iic", "punc_ct-transformer_zh-cn-common-vocab272727-pytorch")
            print(f"Loading local models from: {models_dir}")
            if self.pipeline_b is None:
                print("[AI] loading pipeline B (Sichuan Dialect with optimized VAD)...")
                # [Task 1] Optimize ASR model loading with explicit VAD kwargs
                self.pipeline_b = self.AutoModel(
                    model=path_b,
                    vad_model=path_vad,
                    vad_kwargs={
                        "max_single_segment_time": 60000,
                        "threshold": 0.3,
                        "min_speech_duration_ms": 250,
                        "speech_pad_ms": 200
                    },
                    device=self.device,
                    disable_update=True,
                )
                print("[AI] pipeline B ready")
            if getattr(self, "punc_model", None) is None:
                try:
                    self.punc_model = self.AutoModel(
                        model=path_punc,
                        device=self.device,
                        disable_update=True,
                    )
                except Exception:
                    self.punc_model = None
        except Exception as e:
            raise RuntimeError(f"Model Init Failed: {str(e)}")

    def _preprocess_audio(self, audio_path):
        wav, sr = self.torchaudio.load(audio_path)
        energy = float(self.np.mean(self.np.abs(wav.numpy()))) if wav.numel() > 0 else 0.0
        if energy < 0.01:
            gain = 0.05 / (energy + 1e-6)
            if gain > 50.0:
                gain = 50.0
            wav = wav * gain
            new_path = audio_path.replace(".wav", "_amp.wav")
            self.torchaudio.save(new_path, wav, sr)
            return new_path
        return audio_path

    def _normalize_speaker_labels(self, results):
        """
        [Fix] 将随机的 speaker ID (如 spk1, spk4) 归一化为标准 spk0, spk1
        """
        if not results:
            return []
            
        # 1. 统计所有出现的说话人 ID，按出现顺序排序
        unique_spks = []
        for item in results:
            s = item['speaker']
            if s not in unique_spks:
                unique_spks.append(s)
        
        # 2. 建立映射表: {'spk1': 'spk0', 'spk4': 'spk1'}
        spk_map = {old_id: f"spk{i}" for i, old_id in enumerate(unique_spks)}
        
        # 3. 替换
        for item in results:
            item['speaker'] = spk_map.get(item['speaker'], item['speaker'])
            
        return results

    def run(self, audio_path):
        if not os.path.exists(audio_path):
            print(json.dumps({"error": f"File not found: {audio_path}"}))
            return

        try:
            print(f"[AI] run on {audio_path}")
            self.init_models()
            print("[AI] models initialized")
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            out_dir = os.path.join(base_dir, "app", "static", "separated", base_name)
            print(f"[AI] diarization out_dir={out_dir}")
            
            # [Requirement 2] Use DiarizationService.separate() as sole source
            segs = self.diarizer.separate(audio_path, out_dir)
            
            # [Requirement 4] Log enhancement
            print(f"[3D-Speaker] segments_count = {len(segs)}", file=sys.stderr)
            
            final_output = []
            for i, seg in enumerate(segs):
                start = float(seg.get("start", 0.0))
                end = float(seg.get("end", start))
                spk = str(seg.get("spk", "spk0"))
                rel_path = seg.get("file")
                
                # [Requirement 4] Log segment details
                print(f"[3D-Speaker] seg {i}: spk={spk}, start={start}, end={end}, file={rel_path}", file=sys.stderr)
                
                # Resolve absolute path for processing
                in_path = rel_path
                if in_path and not os.path.isabs(in_path):
                    if in_path.replace("\\", "/").startswith("static/"):
                         # Old compatibility
                        in_path = os.path.join(base_dir, "app", in_path)
                    else:
                        in_path = os.path.join(base_dir, in_path)
                
                # [Requirement 3] ASR per segment
                # Preprocess (volume normalization)
                proc_path = self._preprocess_audio(in_path) if in_path else in_path
                
                text_content = ""
                if self.pipeline_b is not None and proc_path:
                    try:
                        # Pass the segment file directly to ASR
                        res_b = self.pipeline_b.generate(input=proc_path)
                        if isinstance(res_b, list) and len(res_b) > 0:
                            text_content = res_b[0].get('text', '')
                            if getattr(self, "punc_model", None) is not None and text_content:
                                try:
                                    punc_res = self.punc_model.generate(input=text_content)
                                    if isinstance(punc_res, list) and len(punc_res) > 0:
                                        text_content = punc_res[0].get('text', text_content)
                                except Exception:
                                    pass
                    except Exception:
                        text_content = ""
                
                cleaned = text_content
                if cleaned:
                    cleaned = re.sub(r'([。？！，、])\1+', r'\1', cleaned)
                    cleaned = cleaned.replace("，。", "。").replace("？。", "？").replace("！。", "！")
                    cleaned = cleaned.lstrip("。？！，、")
                
                # Use the relative path provided by diarizer for the final output
                final_output.append({
                    "speaker": spk,
                    "text": cleaned,
                    "start": start,
                    "end": end,
                    "path": rel_path
                })

            # [Requirement] Disable merging to preserve 3D-Speaker segmentation
            merged = final_output

            if getattr(self, "punc_model", None) is not None:
                for m in merged:
                    if m["text"]:
                        try:
                            punc_res2 = self.punc_model.generate(input=m["text"])
                            if isinstance(punc_res2, list) and len(punc_res2) > 0:
                                t2 = punc_res2[0].get('text', m["text"]) 
                                # 再次清洗
                                t2 = re.sub(r'([。？！，、])\1+', r'\1', t2)
                                t2 = t2.replace("，。", "。").replace("？。", "？").replace("！。", "！")
                                t2 = t2.lstrip("。？！，、")
                                m["text"] = t2
                        except Exception:
                            pass

            # [新增] 归一化标签，确保输出总是 spk0, spk1...
            merged = self._normalize_speaker_labels(merged)

            print(json.dumps(merged, ensure_ascii=False))
            print("[AI] done")
            return merged

        except Exception as e:
            # 打印错误堆栈到 stderr，打印 JSON 错误到 stdout
            traceback.print_exc(file=sys.stderr)
            print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: python ai_service.py <audio_path>"}))
    else:
        if platform.system().lower().startswith("win"):
            try:
                from app.utils.wsl_bridge import run_in_wsl
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                audio_arg = sys.argv[1]
                if not os.path.isabs(audio_arg):
                    audio_arg = os.path.join(base_dir, audio_arg)
                rc, out, err = run_in_wsl("app/services/ai_service.py", audio_arg)
                if out:
                    print(out.strip())
                if rc != 0:
                    print(json.dumps({"error": (err.strip() or out.strip() or "WSL execution failed")}))
                sys.exit(0)
            except Exception as _e:
                print(json.dumps({"error": str(_e)}))
                sys.exit(1)
        runner = AIServiceRunner()
        runner.run(sys.argv[1])
