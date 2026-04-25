import os
import numpy as np
import sqlite3

db_path = "voice_data.db"
project_root = os.getcwd()

def get_score():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. 获取声纹库 ID 为 1 的 embedding 路径
    cursor.execute("SELECT person_name, embedding_path FROM voice_print WHERE id = 1")
    row_vp = cursor.fetchone()
    if not row_vp:
        print("Error: VoicePrint ID 1 not found.")
        return
    vp_name, vp_path = row_vp
    
    # 2. 获取记录 ID 为 114 的 spk0 embedding 路径
    cursor.execute("SELECT embedding_path FROM record_speaker WHERE record_id = 114 AND raw_spk = 'spk0'")
    row_rs = cursor.fetchone()
    if not row_rs:
        # 如果没搜到 114，搜搜最近的一条记录
        cursor.execute("SELECT record_id, raw_spk, embedding_path FROM record_speaker ORDER BY id DESC LIMIT 1")
        row_rs = cursor.fetchone()
        if not row_rs:
            print("Error: No record speakers found.")
            return
        print(f"Note: Record 114 spk0 not found, using latest record {row_rs[0]} {row_rs[1]}")
        rs_path = row_rs[2]
    else:
        rs_path = row_rs[0]
    
    conn.close()
    
    # 3. 加载并计算
    vp_abs = os.path.join(project_root, "app", vp_path)
    rs_abs = os.path.join(project_root, "app", rs_path)
    
    if not os.path.exists(vp_abs):
        print(f"Error: VP file missing at {vp_abs}")
        return
    if not os.path.exists(rs_abs):
        print(f"Error: Record speaker file missing at {rs_abs}")
        return
        
    emb_vp = np.load(vp_abs)
    emb_rs = np.load(rs_abs)
    
    score = np.dot(emb_vp, emb_rs) / (np.linalg.norm(emb_vp) * np.linalg.norm(emb_rs) + 1e-6)
    
    print("-" * 50)
    print(f"【对比结果】")
    print(f"声纹库目标: {vp_name} (ID: 1)")
    print(f"录音待检方: spk0 (Record: 114)")
    print(f"余弦相似度 (Score): {score:.6f}")
    print("-" * 50)

if __name__ == "__main__":
    get_score()
