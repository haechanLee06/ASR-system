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
        
    conn.close()
    print("Migration complete.")

if __name__ == "__main__":
    migrate()