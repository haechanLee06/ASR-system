import sqlite3
import os

def migrate():
    # Find the database file
    possible_paths = [
        "voice_data.db",
        os.path.join("instance", "voice.db"),
    ]
    
    target_db = None
    for p in possible_paths:
        if os.path.exists(p):
            target_db = p
            break
            
    if not target_db:
        print("Database not found. Skipping migration.")
        return

    print(f"Migrating database at {target_db}...")
    conn = sqlite3.connect(target_db)
    cursor = conn.cursor()
    
    # Check if columns exist
    cursor.execute("PRAGMA table_info(audio_record)")
    columns = [row[1] for row in cursor.fetchall()]
    
    if "error_message" not in columns:
        print("Adding error_message column...")
        cursor.execute("ALTER TABLE audio_record ADD COLUMN error_message TEXT")
        conn.commit()
    else:
        print("error_message column already exists.")
    
    if "current_stage" not in columns:
        print("Adding current_stage column...")
        cursor.execute("ALTER TABLE audio_record ADD COLUMN current_stage TEXT DEFAULT '等待处理'")
        conn.commit()
    else:
        print("current_stage column already exists.")
        
    # --- 下面是我们刚刚新增的 AI 总结字段迁移逻辑 ---
    if "llm_summary" not in columns:
        print("Adding llm_summary column...")
        cursor.execute("ALTER TABLE audio_record ADD COLUMN llm_summary TEXT")
        conn.commit()
    else:
        print("llm_summary column already exists.")

    # --- 2024-04-22 新增：记录标题与最后修改时间 ---
    if "title" not in columns:
        print("Adding title column...")
        cursor.execute("ALTER TABLE audio_record ADD COLUMN title VARCHAR(256)")
        conn.commit()
    else:
        print("title column already exists.")

    if "updated_at" not in columns:
        print("Adding updated_at column...")
        cursor.execute("ALTER TABLE audio_record ADD COLUMN updated_at DATETIME")
        # 为现有记录初始化 updated_at 为 upload_time
        cursor.execute("UPDATE audio_record SET updated_at = upload_time WHERE updated_at IS NULL")
        conn.commit()
    else:
        print("updated_at column already exists.")
        
    conn.close()
    print("Migration complete.")

if __name__ == "__main__":
    migrate()