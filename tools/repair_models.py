#!/usr/bin/env python
"""
修复模型文件损坏问题的工具
"""
import os
import sys
import torch
import hashlib
from pathlib import Path

def validate_model_file(model_path):
    """
    验证模型文件是否有效
    """
    try:
        # 尝试加载模型文件
        state_dict = torch.load(model_path, map_location='cpu')
        print(f"✓ Model file {model_path} is valid")
        return True
    except Exception as e:
        print(f"✗ Model file {model_path} is corrupted: {e}")
        return False

def calculate_file_hash(filepath):
    """
    计算文件的MD5哈希值
    """
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"Error calculating hash for {filepath}: {e}")
        return None

def repair_corrupted_models():
    """
    检查并修复损坏的模型文件
    """
    base_dir = Path(__file__).parent.parent
    models_dir = base_dir / "models"
    
    # 定义需要检查的模型路径
    model_paths = [
        models_dir / "lukeewin01" / "paraformer-large-sichuan-offline" / "model.pt",
        models_dir / "iic" / "speech_fsmn_vad_zh-cn-16k-common-pytorch" / "model.pt",
        models_dir / "iic" / "punc_ct-transformer_zh-cn-common-vocab272727-pytorch" / "model.pt"
    ]
    
    corrupted_models = []
    
    print("Checking model files...")
    for model_path in model_paths:
        if not model_path.exists():
            print(f"✗ Model file does not exist: {model_path}")
            corrupted_models.append(model_path)
        elif not validate_model_file(str(model_path)):
            print(f"✗ Model file is corrupted: {model_path}")
            corrupted_models.append(model_path)
        else:
            file_hash = calculate_file_hash(model_path)
            print(f"  File: {model_path.name}, Size: {model_path.stat().st_size} bytes, Hash: {file_hash[:8]}...")
    
    if corrupted_models:
        print(f"\nFound {len(corrupted_models)} corrupted model(s). Attempting to repair...")
        
        for corrupted_model in corrupted_models:
            print(f"\nTo repair {corrupted_model}:")
            print(f"  1. Remove the corrupted model: rm '{corrupted_model}'")
            print(f"  2. Run the application again to trigger redownload")
            print(f"  3. Or manually download the model from the FunASR modelscope repository")
            
            # 提供重命名选项，以便用户可以备份当前损坏的文件
            backup_path = str(corrupted_model) + ".corrupted"
            print(f"  4. Optionally rename to backup: mv '{corrupted_model}' '{backup_path}'")
    else:
        print("\n✓ All model files are valid!")
        
    return len(corrupted_models) == 0

def force_redownload_models():
    """
    强制重新下载所有模型
    """
    base_dir = Path(__file__).parent.parent
    models_dir = base_dir / "models"
    
    model_paths = [
        models_dir / "lukeewin01" / "paraformer-large-sichuan-offline",
        models_dir / "iic" / "speech_fsmn_vad_zh-cn-16k-common-pytorch",
        models_dir / "iic" / "punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
    ]
    
    print("Preparing to re-download models...")
    for model_path in model_paths:
        if model_path.exists():
            print(f"Removing {model_path}...")
            import shutil
            shutil.rmtree(model_path)
    
    print("\nModels removed. Run your application again to trigger fresh downloads.")

if __name__ == "__main__":
    print("Model Repair Tool for ASR System")
    print("="*50)
    
    if len(sys.argv) > 1 and sys.argv[1] == "force":
        print("Force re-download mode selected.")
        force_redownload_models()
    else:
        print("Validation mode selected.")
        success = repair_corrupted_models()
        
        if not success:
            print("\n" + "="*50)
            print("Recommendation:")
            print("Run with 'force' argument to remove and re-download all models:")
            print(f"  python {__file__} force")