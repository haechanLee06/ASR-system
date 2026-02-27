#!/usr/bin/env python
"""
使用ModelScope下载模型文件的工具
"""
import os
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model

def download_models():
    """
    下载所需的所有模型
    """
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # 模型信息：(模型ID, 本地目录名, 本地子目录名)
    models_to_download = [
        ("lukeewin01/paraformer-large-sichuan-offline", "lukeewin01", "paraformer-large-sichuan-offline"),
        ("iic/speech_fsmn_vad_zh-cn-16k-common-pytorch", "iic", "speech_fsmn_vad_zh-cn-16k-common-pytorch"),
        ("iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch", "iic", "punc_ct-transformer_zh-cn-common-vocab272727-pytorch")
    ]
    
    for model_id, parent_dir, child_dir in models_to_download:
        print(f"开始下载模型: {model_id}")
        
        # 创建目标目录
        target_dir = os.path.join(models_dir, parent_dir, child_dir)
        os.makedirs(target_dir, exist_ok=True)
        
        try:
            # 使用ModelScope下载模型
            model_dir = snapshot_download(
                model_id=model_id,
                cache_dir=models_dir,
                local_dir=target_dir,
                revision='master'
            )
            print(f"模型 {model_id} 已下载到: {model_dir}")
        except Exception as e:
            print(f"下载模型 {model_id} 时出错: {str(e)}")
            continue
    
    print("所有模型下载完成！")

if __name__ == "__main__":
    try:
        from modelscope.hub.snapshot_download import snapshot_download
        print("开始下载模型...")
        download_models()
    except ImportError as e:
        print(f"缺少必要的库，请先安装: pip install modelscope")
        print(f"导入错误详情: {e}")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()