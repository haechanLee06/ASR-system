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
    def __init__(self, asr_only=False):
        self.AutoModel, self.torch, self.torchaudio, self.np = setup_env()
        self.device = "cuda:0" if self.torch.cuda.is_available() else "cpu"
        self.pipeline_b = None
        self.asr_only = asr_only
        if not self.asr_only:
            base_dir_mod = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if base_dir_mod not in sys.path:
                sys.path.append(base_dir_mod)
            from app.services.diarization import DiarizationService
            self.diarizer = DiarizationService()

    def init_models(self):
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            models_dir = os.path.join(base_dir, "models")
            print(f"[AI] base_dir={base_dir}", file=sys.stderr)
            print(f"[AI] models_dir={models_dir}", file=sys.stderr)
            if not os.path.exists(models_dir):
                raise RuntimeError(f"Models folder not found at {models_dir}")
            path_b = os.path.join(models_dir, "lukeewin01", "paraformer-large-sichuan-offline")
            path_vad = os.path.join(models_dir, "iic", "speech_fsmn_vad_zh-cn-16k-common-pytorch")
            path_punc = os.path.join(models_dir, "iic", "punc_ct-transformer_zh-cn-common-vocab272727-pytorch")
            print(f"Loading local models from: {models_dir}", file=sys.stderr)
            if self.pipeline_b is None:
                print("[AI] loading pipeline B (Sichuan Dialect with optimized VAD)...", file=sys.stderr)
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
                print("[AI] pipeline B ready", file=sys.stderr)
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
        if not results:
            return []
        unique_spks = []
        for item in results:
            s = item['speaker']
            if s not in unique_spks:
                unique_spks.append(s)
        spk_map = {old_id: f"spk{i}" for i, old_id in enumerate(unique_spks)}
        for item in results:
            item['speaker'] = spk_map.get(item['speaker'], item['speaker'])
        return results

    def run(self, audio_path):
        if not os.path.exists(audio_path):
            print(json.dumps({"error": f"File not found: {audio_path}"}))
            return

        try:
            print(f"[AI] run on {audio_path}", file=sys.stderr)
            self.init_models()
            print("[AI] models initialized", file=sys.stderr)
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            if self.asr_only:
                # 仅执行 ASR (隔离加载)
                proc_path = self._preprocess_audio(audio_path)
                text_content = ""
                if self.pipeline_b is not None and proc_path:
                    try:
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
                
                output = [{"text": cleaned, "path": audio_path, "speaker": "spk0", "start": 0.0, "end": 0.0}]
                print(json.dumps(output, ensure_ascii=False))
                return output

            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            out_dir = os.path.join(base_dir, "app", "static", "separated", base_name)
            print(f"[AI] diarization out_dir={out_dir}", file=sys.stderr)
            
            segs, spk_embs = self.diarizer.separate(audio_path, out_dir)
            print(f"[3D-Speaker] segments_count = {len(segs)}", file=sys.stderr)
            
            final_output = []
            for i, seg in enumerate(segs):
                start = float(seg.get("start", 0.0))
                end = float(seg.get("end", start))
                spk = str(seg.get("spk", "spk0"))
                rel_path = seg.get("file")
                
                print(f"[3D-Speaker] seg {i}: spk={spk}, start={start}, end={end}, file={rel_path}", file=sys.stderr)
                
                in_path = rel_path
                if in_path and not os.path.isabs(in_path):
                    if in_path.replace("\\", "/").startswith("static/"):
                        in_path = os.path.join(base_dir, "app", in_path)
                    else:
                        in_path = os.path.join(base_dir, in_path)
                
                proc_path = self._preprocess_audio(in_path) if in_path else in_path
                
                text_content = ""
                if self.pipeline_b is not None and proc_path:
                    try:
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
                
                final_output.append({
                    "speaker": spk,
                    "text": cleaned,
                    "start": start,
                    "end": end,
                    "path": rel_path,
                    "embedding": spk_embs.get(spk)
                })

            merged = final_output

            if getattr(self, "punc_model", None) is not None:
                for m in merged:
                    if m["text"]:
                        try:
                            punc_res2 = self.punc_model.generate(input=m["text"])
                            if isinstance(punc_res2, list) and len(punc_res2) > 0:
                                t2 = punc_res2[0].get('text', m["text"]) 
                                t2 = re.sub(r'([。？！，、])\1+', r'\1', t2)
                                t2 = t2.replace("，。", "。").replace("？。", "？").replace("！。", "！")
                                t2 = t2.lstrip("。？！，、")
                                m["text"] = t2
                        except Exception:
                            pass

            merged = self._normalize_speaker_labels(merged)
            print(json.dumps(merged, ensure_ascii=False))
            print("[AI] done", file=sys.stderr)
            return merged

        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: python ai_service.py <audio_path> [--asr_only]"}))
    else:
        asr_only = "--asr_only" in sys.argv
        
        # Gather all valid positional arguments
        audio_args = [arg for arg in sys.argv[1:] if arg != "--asr_only"]
        
        if platform.system().lower().startswith("win"):
            try:
                from app.utils.wsl_bridge import run_in_wsl
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                # Process the first argument normally
                first_arg = audio_args[0]
                if not os.path.isabs(first_arg):
                    first_arg = os.path.join(base_dir, first_arg)
                    
                # Append subsequent args
                extra_args = audio_args[1:]
                if asr_only:
                    extra_args.append("--asr_only")
                res = run_in_wsl("app/services/ai_service.py", first_arg, *extra_args)
                print(json.dumps(res, ensure_ascii=False))
                sys.exit(0)
            except Exception as _e:
                print(json.dumps({"error": str(_e)}))
                sys.exit(1)
                
        runner = AIServiceRunner(asr_only=asr_only)
        final_results = []
        for audio_arg in audio_args:
            res_chunk = runner.run(audio_arg)
            if res_chunk:
                final_results.extend(res_chunk)
        
        print(json.dumps(final_results, ensure_ascii=False))

