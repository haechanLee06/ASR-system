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


def _extract_json_str(raw: str) -> str:
    """
    层先1：从模型输出中提取有效 JSON 字符串。
    兼容模型将内容包在 ```json ... ``` 或 ``` ... ``` 中的情况。
    """
    raw = raw.strip()
    # 去掉 markdown 代码块包裹
    if raw.startswith("```"):
        lines = raw.splitlines()
        # 移除首行 (```json 或 ```) 和尾行 (```)
        inner = []
        for line in lines[1:]:
            if line.strip() == "```":
                break
            inner.append(line)
        raw = "\n".join(inner).strip()
    return raw


def _ensure_structure(data: dict) -> dict:
    """
    层先2：确保返回字典具备前端期望的完整结构。
    如果模型输出的 JSON 缺少某些键，填充空字符串。
    """
    EMPTY_ROLE = {
        "role_label": "",
        "emotional_state": "",
        "hidden_intent": "",
        "VFA_analysis": {
            "viewpoint": "",
            "facts": [],
            "deep_analysis": ""
        }
    }
    EMPTY_STRUCTURE = {
        "summary": {
            "overview": {
                "scene_type": "",
                "detailed_summary": ""
            },
            "analysis_breakdown": {
                "role_0": EMPTY_ROLE.copy(),
                "role_1": EMPTY_ROLE.copy()
            }
        }
    }

    # 若模型直接输出了 summary 内部内容而非整个包裹结构，则手动包裹
    if "overview" in data and "summary" not in data:
        data = {"summary": data}
    elif "analysis_breakdown" in data and "summary" not in data:
        data = {"summary": data}

    summary = data.get("summary", {})
    if not isinstance(summary, dict):
        return EMPTY_STRUCTURE

    # 层级填充
    overview = summary.get("overview", {})
    if not isinstance(overview, dict):
        overview = {}
    overview.setdefault("scene_type", "")
    overview.setdefault("detailed_summary", "")
    summary["overview"] = overview

    breakdown = summary.get("analysis_breakdown", {})
    if not isinstance(breakdown, dict):
        breakdown = {}
    for role_key in ("role_0", "role_1"):
        role = breakdown.get(role_key, {})
        if not isinstance(role, dict):
            role = {}
        role.setdefault("role_label", "")
        role.setdefault("emotional_state", "")
        role.setdefault("hidden_intent", "")
        vfa = role.get("VFA_analysis", {})
        if not isinstance(vfa, dict):
            vfa = {}
        vfa.setdefault("viewpoint", "")
        vfa.setdefault("facts", [])
        vfa.setdefault("deep_analysis", "")
        role["VFA_analysis"] = vfa
        breakdown[role_key] = role
    summary["analysis_breakdown"] = breakdown
    data["summary"] = summary

    return data


def request_local_llm(transcript_text):
    """
    专门调用本地 Ollama 微调大模型 (qwen-sichuan-psych) 进行总结。
    采用流式读取（stream=True），逐 chunk 拼接，避免 300s 整体阻塞超时。
    """
    url = "http://127.0.0.1:11434/api/chat"

    system_prompt = (
        "你是一个专业的心理和行为分析专家。请阅读提供的对话记录，并进行深度侧写。\n"
        "【严格警告】：你必须严格按照以下 JSON 模板输出数据，绝对不能更改键值名（Key），不能增减任何层级。"
        "不要输出任何解释性文字或 Markdown 标记（如 ```json）。\n\n"
        "必须输出的 JSON 结构模板：\n"
        "{\n"
        '  "summary": {\n'
        '    "overview": {\n'
        '      "scene_type": "用不超过10个字概括场景",\n'
        '      "detailed_summary": "用一两句话概括事情的起因、经过和结果"\n'
        "    },\n"
        '    "analysis_breakdown": {\n'
        '      "role_0": {\n'
        '        "role_label": "角色0的身份定性",\n'
        '        "emotional_state": "情绪动线",\n'
        '        "hidden_intent": "该角色的深层隐性意图",\n'
        '        "VFA_analysis": {\n'
        '          "viewpoint": "该角色的核心观点",\n'
        '          "facts": ["事实论据1", "事实论据2"],\n'
        '          "deep_analysis": "深度剖析该角色的逻辑与心理动因"\n'
        "        }\n"
        "      },\n"
        '      "role_1": {\n'
        '        "role_label": "角色1的身份定性",\n'
        '        "emotional_state": "情绪动线",\n'
        '        "hidden_intent": "该角色的深层隐性意图",\n'
        '        "VFA_analysis": {\n'
        '          "viewpoint": "该角色的核心观点",\n'
        '          "facts": ["事实论据1", "事实论据2"],\n'
        '          "deep_analysis": "深度剖析该角色的逻辑与心理动因"\n'
        "        }\n"
        "      }\n"
        "    }\n"
        "  }\n"
        "}\n\n"
        '注意：角色键名必须是 "role_0" 和 "role_1"（绝对不能用 role_A 或 role_B）。'
    )

    payload = {
        "model": "qwen-sichuan-psych",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "以下是需要分析的对话内容：\n\n" + transcript_text}
        ],
        "stream": True,  # 流式模式：逐 chunk 读取，避免单次阻塞 300s
    }

    content_str = ""
    try:
        # timeout=(connect_timeout, read_timeout)
        # read_timeout=120 是每两次收到数据之间的最大间隔，不是总时长
        with requests.post(url, json=payload, stream=True, timeout=(10, 120)) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line.decode("utf-8"))
                except (json.decoder.JSONDecodeError, UnicodeDecodeError):
                    continue

                delta = chunk.get("message", {}).get("content", "")
                if delta:
                    content_str += delta

                # Ollama 在推理完成时会将 done 字段置为 True
                if chunk.get("done", False):
                    break

        print(f"[LLM] Streaming complete, total chars: {len(content_str)}")

        if not content_str:
            print("[LLM] Warning: empty content from model, returning empty structure")
            return _ensure_structure({})

        # 层级1：去掉可能存在的 Markdown 包裹
        json_str = _extract_json_str(content_str)

        # 层级2：JSON 解析，失败时尝试从内容中抽取第一个 { ... } 块
        try:
            parsed = json.loads(json_str)
        except json.decoder.JSONDecodeError:
            import re
            match = re.search(r'\{.*\}', content_str, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                    print("[LLM] Warning: used regex fallback to extract JSON")
                except json.decoder.JSONDecodeError as e:
                    print(f"[LLM] JSON decode failed after regex fallback. Raw: {content_str[:300]}")
                    raise RuntimeError(f"大模型返回的格式错误，非有效 JSON: {e}")
            else:
                print(f"[LLM] No JSON found in response. Raw: {content_str[:300]}")
                raise RuntimeError("大模型未返回任何 JSON 结构")

        # 层级3：字段完整性袥底
        return _ensure_structure(parsed)

    except requests.exceptions.ConnectionError as e:
        print(f"[LLM] Cannot connect to Ollama: {e}")
        raise RuntimeError("无法连接到 Ollama，请确认服务是否已在 127.0.0.1:11434 启动")
    except requests.exceptions.Timeout as e:
        print(f"[LLM] Request timed out: {e}")
        raise RuntimeError("大模型推理超时，请检查 Ollama 状态")
    except requests.exceptions.RequestException as e:
        print(f"[LLM] Request failed: {e}")
        raise RuntimeError(f"请求大模型失败: {e}")
