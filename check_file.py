from modelscope.hub.snapshot_download import snapshot_download
from pathlib import Path

# 配置路径（保持和你之前下载的一样）
TARGET_ROOT = Path("/18t/data/home/panyq/llm_data/dataset")
REPO_ID = 'pengzhendong/WenetSpeech-Chuan'

def get_metadata_only():
    dataset_name = REPO_ID.split('/')[-1]
    local_dir = TARGET_ROOT / dataset_name
    
    print(f"🚀 开始精准搜索标注文件...")
    print(f"目标目录: {local_dir}")
    
    try:
        snapshot_download(
            REPO_ID,
            repo_type='dataset',
            local_dir=str(local_dir),
            # 关键参数：忽略所有的 .tar 文件
            # 这样它就不会去校验你那 600G 的数据，只下载漏掉的小文件
            ignore_file_pattern=['*.tar'], 
        )
        print("\n✅ 补全完成！")
        
    except Exception as e:
        print(f"❌ 出错: {e}")

if __name__ == "__main__":
    get_metadata_only()