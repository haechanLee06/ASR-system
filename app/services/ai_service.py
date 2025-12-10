import sys
import os
import json
import traceback
import logging

# 屏蔽繁杂的日志
logging.getLogger('modelscope').setLevel(logging.ERROR)
logging.getLogger('funasr').setLevel(logging.ERROR)

def setup_env():
    # 尝试导入 funasr
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
        self.pipeline_a = None
        self.pipeline_b = None

    def init_models(self):
        """初始化双模型"""
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            models_dir = os.path.join(base_dir, "models")
            print(f"[AI] base_dir={base_dir}")
            print(f"[AI] models_dir={models_dir}")
            if not os.path.exists(models_dir):
                raise RuntimeError(f"Models folder not found at {models_dir}")

            path_a = os.path.join(models_dir, "iic", "speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
            path_b = os.path.join(models_dir, "lukeewin01", "paraformer-large-sichuan-offline")
            path_vad = os.path.join(models_dir, "iic", "speech_fsmn_vad_zh-cn-16k-common-pytorch")
            path_punc = os.path.join(models_dir, "iic", "punc_ct-transformer_zh-cn-common-vocab272727-pytorch")
            path_spk = os.path.join(models_dir, "iic", "speech_campplus_sv_zh-cn_16k-common")

            print(f"Loading local models from: {models_dir}")

            if self.pipeline_a is None:
                print("[AI] loading pipeline A...")
                self.pipeline_a = self.AutoModel(
                    model=path_a,
                    device=self.device,
                    vad_model=path_vad,
                    vad_kwargs={"max_single_segment_time": 60000},
                    punc_model=path_punc,
                    spk_model=path_spk,
                    disable_update=True,
                )
                print("[AI] pipeline A ready")

            if self.pipeline_b is None:
                print("[AI] loading pipeline B...")
                self.pipeline_b = self.AutoModel(
                    model=path_b,
                    device=self.device,
                    disable_update=True,
                )
                print("[AI] pipeline B ready")
        except Exception as e:
            raise RuntimeError(f"Model Init Failed: {str(e)}")

    def cut_audio(self, wav, sr, start, end):
        """内存音频切片"""
        # wav shape: [channels, time]
        if sr != 16000:
            T = self.torchaudio.transforms
            resampler = T.Resample(sr, 16000)
            wav = resampler(wav)
            sr = 16000
        
        # 转单声道
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        
        wav = wav.squeeze(0) # [time]
        
        total_frames = wav.shape[0]
        start_frame = int(start * sr)
        end_frame = int(end * sr)
        
        start_frame = max(0, min(start_frame, total_frames))
        end_frame = max(0, min(end_frame, total_frames))
        
        if end_frame <= start_frame:
            return None
            
        # 返回 numpy 数组，供 FunASR 输入
        return wav[start_frame:end_frame].detach().cpu().numpy()

    def run(self, audio_path):
        if not os.path.exists(audio_path):
            print(json.dumps({"error": f"File not found: {audio_path}"}))
            return

        try:
            print(f"[AI] run on {audio_path}")
            self.init_models()
            print("[AI] models initialized")
            
            # 1. Pipeline A 全文推理
            # return_spk_res=True 激活说话人识别
            res_a = self.pipeline_a.generate(
                input=audio_path,
                batch_size_s=300,
                sentence_timestamp=True,
                return_spk_res=True
            )
            print("[AI] pipeline A inference done")
            
            # 解析结果
            raw_segments = []
            if isinstance(res_a, list) and len(res_a) > 0:
                item = res_a[0]
                if 'sentence_info' in item:
                    for sent in item['sentence_info']:
                        raw_segments.append({
                            "start": sent['start'] / 1000.0, # ms -> s
                            "end": sent['end'] / 1000.0,
                            "spk": f"spk{sent.get('spk', 0)}"
                        })
            print(f"[AI] segments={len(raw_segments)}")
            
            # 如果没识别出句子，兜底处理
            if not raw_segments:
                raw_segments.append({"start": 0.0, "end": 0.0, "spk": "spk0"})

            # 2. 准备音频数据进行切片
            wav, sr = self.torchaudio.load(audio_path)
            print(f"[AI] audio loaded sr={sr}")
            
            final_output = []
            
            # 3. Pipeline B 逐句精修
            for seg in raw_segments:
                start, end, spk = seg['start'], seg['end'], seg['spk']
                print(f"[AI] process seg start={start} end={end} spk={spk}")
                
                # 如果是兜底数据(0,0)，直接识别全文
                if start == 0.0 and end == 0.0:
                    audio_chunk = wav.mean(dim=0).detach().cpu().numpy() if wav.shape[0] > 1 else wav.squeeze().detach().cpu().numpy()
                else:
                    audio_chunk = self.cut_audio(wav, sr, start, end)
                
                text_content = ""
                if audio_chunk is not None and len(audio_chunk) > 0:
                    try:
                        # 传入 numpy 数组进行识别
                        res_b = self.pipeline_b.generate(input=audio_chunk)
                        print("[AI] pipeline B inference done")
                        # 解析 B 模型结果
                        if isinstance(res_b, list) and len(res_b) > 0:
                            text_content = res_b[0].get('text', '')
                    except Exception as e:
                        text_content = f"[Err: {str(e)}]"
                
                final_output.append({
                    "speaker": spk,
                    "text": text_content,
                    "start": start,
                    "end": end
                })

            # 4. 输出标准 JSON
            print(json.dumps(final_output, ensure_ascii=False))
            print("[AI] done")

        except Exception as e:
            # 打印错误堆栈到 stderr，打印 JSON 错误到 stdout
            traceback.print_exc(file=sys.stderr)
            print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: python ai_service.py <audio_path>"}))
    else:
        runner = AIServiceRunner()
        runner.run(sys.argv[1])
