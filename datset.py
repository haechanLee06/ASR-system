import json
import os
import random
from pathlib import Path
from tqdm import tqdm

# ================= 配置路径 =================
# 1. 原始数据路径
JSONL_PATH = Path("/18t/data/home/panyq/llm_data/dataset/WenetSpeech-Chuan/text.jsonl")
AUDIO_ROOT = Path("/18t/data/home/panyq/llm_data/dataset/WenetSpeech-Chuan/raw_wav")

# 2. 输出路径 (会自动创建 train 和 val 文件夹)
OUTPUT_ROOT = Path("/18t/data/home/panyq/llm_data/dataset/funasr_finetune")

# 3. 验证集数量 (从总数里切出多少条用来测试效果？建议 500-1000)
VAL_COUNT = 1000
# ===========================================

def main():
    if not OUTPUT_ROOT.exists():
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("🚀 正在扫描 150GB+ 的音频文件 (可能需要一分钟)...")
    # 建立 {文件名: 绝对路径} 索引
    wav_map = {f.name: str(f.absolute()) for f in AUDIO_ROOT.rglob("*.wav")}
    print(f"✅ 硬盘里找到了 {len(wav_map)} 个音频文件。")

    print("🚀 正在匹配标注并构建数据集...")
    data_list = [] # 暂存所有匹配成功的数据

    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            try:
                entry = json.loads(line)
                filename = entry.get('filename')
                utt_id = entry.get('utt')
                content = entry.get('text')

                # 只有硬盘里有的才算数
                if filename in wav_map:
                    full_path = wav_map[filename]
                    # 过滤掉过短的音频 (可选，防止报错)
                    # 格式: (ID, 路径, 文本)
                    data_list.append((utt_id, full_path, content))
            except:
                continue

    total = len(data_list)
    print(f"\n📊 匹配完成！共找到有效数据: {total} 条")

    if total < VAL_COUNT + 100:
        print("❌ 数据太少，不够切分验证集！请检查解压是否成功。")
        return

    # 随机打乱
    print("🎲 正在打乱数据并切分...")
    random.shuffle(data_list)

    # 切分验证集和训练集
    val_data = data_list[:VAL_COUNT]
    train_data = data_list[VAL_COUNT:]

    # 写入文件的辅助函数
    def write_files(subset_name, dataset):
        folder = OUTPUT_ROOT / subset_name
        folder.mkdir(parents=True, exist_ok=True)
        
        scp_path = folder / "wav.scp"
        text_path = folder / "text"
        
        with open(scp_path, "w", encoding="utf-8") as f_scp, \
             open(text_path, "w", encoding="utf-8") as f_text:
            for utt, path, txt in dataset:
                f_scp.write(f"{utt} {path}\n")
                f_text.write(f"{utt} {txt}\n")
        
        print(f"   ✅ 已生成 {subset_name}: {len(dataset)} 条 -> {folder}")

    write_files("val", val_data)
    write_files("train", train_data)

    print("\n" + "="*30)
    print(f"🎉 数据准备完毕！")
    print(f"📂 训练数据位于: {OUTPUT_ROOT}/train")
    print(f"📂 验证数据位于: {OUTPUT_ROOT}/val")
    print("="*30)

if __name__ == "__main__":
    main()