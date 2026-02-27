import os
import tarfile
from pathlib import Path
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 压缩包所在目录
TAR_DIR = Path("/18t/data/home/panyq/llm_data/dataset/WenetSpeech-Chuan/audios")

# 2. 解压目标目录
EXTRACT_DIR = Path("/18t/data/home/panyq/llm_data/dataset/WenetSpeech-Chuan/raw_wav")

# 3. ⚠️ 关键设置：你想解压多少个包？
# 设置为 None 表示解压所有 (56个)
# 设置为 10 表示只解压前 10 个 (建议先设这个)
LIMIT = 15  
# ===========================================

def unzip_data():
    if not EXTRACT_DIR.exists():
        EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 获取所有 tar 文件并排序，确保顺序
    tar_files = sorted(list(TAR_DIR.glob("*.tar")))
    
    if not tar_files:
        print("❌ 没找到 .tar 文件！")
        return

    # 应用数量限制
    if LIMIT is not None:
        target_files = tar_files[:LIMIT]
        print(f"🎯 策略：只解压前 {LIMIT} 个包 (总共 {len(tar_files)} 个)")
    else:
        target_files = tar_files
        print(f"🎯 策略：全量解压所有 {len(tar_files)} 个包 (这将耗费较长时间)")

    print(f"📂 解压目标路径: {EXTRACT_DIR}")
    print("-" * 30)

    # 遍历解压
    for i, tar_path in enumerate(target_files):
        print(f"[{i+1}/{len(target_files)}] 正在解压: {tar_path.name} ...")
        try:
            with tarfile.open(tar_path, "r:") as tar:
                # 这一步比较耗时，我们不显示详细进度条以免刷屏，只显示文件进度
                tar.extractall(path=EXTRACT_DIR)
        except Exception as e:
            print(f"⚠️ 解压 {tar_path.name} 失败: {e}")
            continue
            
    print("-" * 30)
    print(f"✅ 解压任务完成！")
    print(f"📊 现在你可以运行 gen_funasr_data.py 来生成训练列表了。")

if __name__ == "__main__":
    # 使用 nohup 运行建议:
    # nohup python -u unzip_flexible.py > unzip.log 2>&1 &
    unzip_data()