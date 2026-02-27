import os
from modelscope.hub.api import HubApi
from modelscope.hub.snapshot_download import snapshot_download
from pathlib import Path

# ================= 配置区域 =================
# 你的目标下载根目录
TARGET_ROOT = Path("/18t/data/home/panyq/llm_data/dataset") 
# 建议单独建一个 dataset 目录，不要混在 models 里，方便管理
# 如果你非要放 models，就把上面改成 "/18t/data/home/panyq/models"

# 你的 Access Token
ACCESS_TOKEN = 'ms-e7fed5b5-220e-4fc9-98bc-fe0ff5fdba15'

# 数据集 ID
REPO_ID = 'pengzhendong/WenetSpeech-Chuan'
# ===========================================

def download_sichuan_dataset():
    # 1. 确保目录存在
    if not TARGET_ROOT.exists():
        print(f"创建目录: {TARGET_ROOT}")
        TARGET_ROOT.mkdir(parents=True, exist_ok=True)

    # 2. 登录 ModelScope
    print("正在登录 ModelScope...")
    api = HubApi()
    try:
        api.login(ACCESS_TOKEN)
        print("登录成功！")
    except Exception as e:
        print(f"登录失败，请检查 Token。错误: {e}")
        return

    # 3. 构造保存路径
    # 最终路径类似: /18t/data/home/panyq/llm_data/dataset/WenetSpeech-Chuan
    dataset_name = REPO_ID.split('/')[-1]
    local_dir = TARGET_ROOT / dataset_name
    
    print(f"准备下载数据集: {REPO_ID}")
    print(f"保存路径: {local_dir}")

    # 4. 开始下载
    try:
        # cache_dir 是缓存路径，local_dir 是最终解压可见的路径
        # 通常建议两者设为一样，或者让 cache_dir 自动管理
        download_path = snapshot_download(
            REPO_ID,
            repo_type='dataset',      # 关键：指定下载的是数据集
            local_dir=str(local_dir), # 指定下载到我们想要的地方
            ignore_file_pattern=[     # 可选：忽略一些不需要的隐藏文件
                '.git', 
                '.gitattributes'
            ]
        )
        print(f"\n✅ 下载完成！数据位于: {download_path}")
        
        # 5. 简单检查一下文件结构
        print("\n文件列表预览:")
        for root, dirs, files in os.walk(download_path):
            level = root.replace(str(download_path), '').count(os.sep)
            indent = ' ' * 4 * (level)
            print(f'{indent}{os.path.basename(root)}/')
            subindent = ' ' * 4 * (level + 1)
            for f in files[:5]: # 只打印前5个文件避免刷屏
                print(f'{subindent}{f}')
            if len(files) > 5:
                print(f'{subindent}...')
            # 只打印第一层子目录，避免太长
            if level > 0: 
                break

    except Exception as e:
        print(f"\n❌ 下载出错: {e}")

if __name__ == "__main__":
    download_sichuan_dataset()