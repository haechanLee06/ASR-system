import os
import json

# 你提供的路径
train_dir = "/18t/data/home/panyq/llm_data/dataset/funasr_finetune/train"
val_dir = "/18t/data/home/panyq/llm_data/dataset/funasr_finetune/val"

def process_data(data_dir, output_file):
    scp_path = os.path.join(data_dir, "wav.scp")
    text_path = os.path.join(data_dir, "text")
    
    if not os.path.exists(scp_path) or not os.path.exists(text_path):
        print(f"Error: 找不到文件 {scp_path} 或 {text_path}")
        return

    # 读取 text
    text_dict = {}
    with open(text_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                text_dict[parts[0]] = parts[1]

    # 读取 wav.scp 并合并
    lines = []
    with open(scp_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                key, wav_path = parts[0], parts[1]
                if key in text_dict:
                    # FunASR 标准格式: source是音频, target是文本
                    entry = {
                        "key": key,
                        "source": wav_path,
                        "target": text_dict[key]
                    }
                    lines.append(json.dumps(entry, ensure_ascii=False))
    
    # 写入 jsonl
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"成功生成: {output_file} (包含 {len(lines)} 条数据)")

# 执行转换
process_data(train_dir, train_dir + ".jsonl")
process_data(val_dir, val_dir + ".jsonl")