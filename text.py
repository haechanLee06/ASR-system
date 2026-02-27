import json

json_path = "/18t/data/home/panyq/models/paraformer-large/tokens.json"
txt_path  = "/18t/data/home/panyq/models/paraformer-large/tokens.txt"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 情况 1：tokens.json 是 list
if isinstance(data, list):
    tokens = data

# 情况 2：tokens.json 是 dict（保险兜底）
elif isinstance(data, dict):
    tokens = [v for k, v in sorted(data.items(), key=lambda x: int(x[0]))]

else:
    raise ValueError("Unsupported tokens.json format")

with open(txt_path, "w", encoding="utf-8") as f:
    for t in tokens:
        f.write(str(t) + "\n")

print("✅ token_list generated:", txt_path)
print("✅ vocab size:", len(tokens))
