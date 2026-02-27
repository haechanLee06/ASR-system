import os
from modelscope.hub.snapshot_download import snapshot_download
from pathlib import Path

# ================= 配置区域 =================
# 1. 模型保存的大本营
MODEL_ROOT = Path("/18t/data/home/panyq/models")

# 2. 我们要选用的“标准版 Paraformer” ID
# 这是一个全能模型：自带 VAD(静音检测) + Punc(标点恢复) + ASR(识别)
MODEL_ID = "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
# ===========================================

def download_base_model():
    # 确保目录存在
    if not MODEL_ROOT.exists():
        MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"🚀 正在下载底座模型: {MODEL_ID} ...")
    print(f"📂 目标路径: {MODEL_ROOT / 'paraformer-large'}")
    
    try:
        # local_dir 指定下载后的文件夹名字，方便引用
        model_dir = snapshot_download(
            MODEL_ID,
            cache_dir=str(MODEL_ROOT), 
            local_dir=str(MODEL_ROOT / "paraformer-large"), 
        )
        print("\n" + "="*30)
        print(f"✅ 模型下载成功！")
        print(f"📍 绝对路径 (训练脚本里填这个): {model_dir}")
        print("="*30)
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")

if __name__ == "__main__":
    download_base_model()