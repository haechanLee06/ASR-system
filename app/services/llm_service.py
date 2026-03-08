import json
import requests

def format_transcript(segments):
    """
    将切片列表整合成完整对话文本和分角色文本。
    供 LLM 服务或前端展示纯文本使用。
    """
    full_text_lines = []
    spk_lines = {}  # e.g. {'spk0': [], 'spk1': []}

    for seg in segments:
        spk = seg.get('speaker', 'unknown')
        text = seg.get('text', '')
        
        if not text:
            continue

        full_text_lines.append(f"{spk}: {text}")

        if spk not in spk_lines:
            spk_lines[spk] = []
        spk_lines[spk].append(text)

    result = {
        "full_transcript": "\n".join(full_text_lines)
    }

    for spk, lines in spk_lines.items():
        result[f"{spk}_transcript"] = "\n".join(lines)

    if "spk0_transcript" not in result:
        result["spk0_transcript"] = ""
    if "spk1_transcript" not in result:
        result["spk1_transcript"] = ""

    return result

def request_local_llm(transcript_text):
    """
    专门调用本地 Ollama 微调大模型 (qwen-sichuan-psych) 进行总结
    """
    url = "http://127.0.0.1:11434/api/chat"
    
    # 构建结构化的 Prompt，强制约束 JSON 输出格式
    system_prompt = """
你是一个专业的心理和行为分析专家。请阅读提供的对话记录，并进行深度侧写。
**【严格警告】**：你必须严格按照以下 JSON 模板输出数据，绝对不能更改键值名（Key），不能增减任何层级。不要输出任何解释性文字或 Markdown 标记（如 ```json）。

**必须输出的 JSON 结构模板：**
{
  "summary": {
    "overview": {
      "scene_type": "用不超过10个字概括场景（如：乘车是否开空调）",
      "detailed_summary": "用一两句话概括事情的起因、经过和结果"
    },
    "analysis_breakdown": {
      "role_0": {
        "role_label": "角色0的身份定性（如：乘客、被监管人员）",
        "emotional_state": "情绪动线",
        "hidden_intent": "该角色的深层隐性意图",
        "VFA_analysis": {
          "viewpoint": "该角色的核心观点",
          "facts": ["事实论据1", "事实论据2"...],
          "deep_analysis": "深度剖析该角色的逻辑与心理动因"
        }
      },
      "role_1": {
        "role_label": "角色1的身份定性（如：司机、监管人员）",
        "emotional_state": "情绪动线",
        "hidden_intent": "该角色的深层隐性意图",
        "VFA_analysis": {
          "viewpoint": "该角色的核心观点",
          "facts": ["事实论据1", "事实论据2",...],
          "deep_analysis": "深度剖析该角色的逻辑与心理动因"
        }
      }
    }
  }
}

注意：角色键名必须是 "role_0" 和 "role_1"（绝对不能用 role_A 或 role_B）。
"""

    payload = {
        "model": "qwen-sichuan-psych",
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": "以下是需要分析的对话内容：\n\n" + transcript_text
            }
        ],
        "stream": False,
        "format": "json" # Ollama 的 json 模式会保证输出的字符串是合法的 JSON
    }
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        res_json = response.json()
        
        content_str = res_json.get("message", {}).get("content", "")
        if not content_str:
            return {}
            
        return json.loads(content_str)
        
    except requests.exceptions.RequestException as e:
        print(f"[LLM] Request failed: {e}")
        raise RuntimeError(f"请求大模型失败: {e}")
    except json.decoder.JSONDecodeError as e:
        print(f"[LLM] JSON decode failed for response: {content_str}")
        raise RuntimeError(f"大模型返回的格式错误，非有效 JSON: {e}")