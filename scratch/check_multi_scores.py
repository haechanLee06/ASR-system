import os
import numpy as np
import sqlite3

db_path = "voice_data.db"
project_root = os.getcwd()

def get_multi_scores():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. 获取声纹库 "2号" 和 "1号" 的特征路径
    cursor.execute("SELECT id, person_name, embedding_path FROM voice_print WHERE person_name IN ('1号', '2号')")
    vps = cursor.fetchall()
    
    # 2. 获取最新一条记录中各角色的特征路径
    cursor.execute("SELECT record_id, raw_spk, embedding_path FROM record_speaker WHERE record_id = (SELECT MAX(record_id) FROM record_speaker)")
    rss = cursor.fetchall()
    
    if not rss:
        print("Error: No record speakers found.")
        return
    
    rec_id = rss[0][0]
    conn.close()
    
    print("=" * 60)
    print(f"【声纹库识别深度诊断报告】")
    print(f"对应录音记录 ID: {rec_id}")
    print("-" * 60)
    
    # 载入声纹库向量
    vp_data = []
    for vid, vname, vpath in vps:
        path = os.path.join(project_root, "app", vpath)
        if os.path.exists(path):
            vp_data.append((vname, np.load(path)))
            
    # 载入录音角色向量
    rs_data = []
    for _, rspk, rpath in rss:
        path = os.path.join(project_root, "app", rpath)
        if os.path.exists(path):
            rs_data.append((rspk, np.load(path)))
            
    # 面向“2号”的专项匹配
    for vname, v_emb in vp_data:
        print(f">>> 声纹库身份: 【{vname}】")
        for rspk, r_emb in rs_data:
            # 计算 Cosine Similarity
            score = np.dot(v_emb, r_emb) / (np.linalg.norm(v_emb) * np.linalg.norm(r_emb) + 1e-6)
            tag = " [ 强匹配! ] " if score >= 0.65 else ""
            print(f"    与 录音端 {rspk} 的匹配得分: {score:.6f}{tag}")
        print("-" * 60)

if __name__ == "__main__":
    get_multi_scores()
