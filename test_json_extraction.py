#!/usr/bin/env python3
"""测试JSON提取函数"""
import json
import re

def _extract_json_from_response(response_text: str) -> dict:
    """从LLM响应中提取JSON对象（增强版）"""

    # 策略1: 尝试直接解析
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # 策略2: 提取markdown代码块中的JSON（```json...```）
    code_block_patterns = [
        r'```json\s*\n(.*?)\n```',  # ```json\n{...}\n```
        r'```\s*\n(\{.*?\})\s*\n```',  # ```\n{...}\n```
        r'```json\s*(.*?)```',  # ```json{...}```（无换行）
    ]

    for pattern in code_block_patterns:
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            try:
                json_str = match.group(1).strip()
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue

    # 策略3: 手动查找markdown代码块中的JSON（括号匹配）
    code_block_match = re.search(r'```(?:json)?\s*(\{)', response_text, re.DOTALL)
    if code_block_match:
        start_pos = code_block_match.start(1)
        brace_count = 0
        for i in range(start_pos, len(response_text)):
            if response_text[i] == '{':
                brace_count += 1
            elif response_text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    try:
                        json_str = response_text[start_pos:i+1]
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        break

    # 策略4: 查找第一个完整的JSON对象（从头开始）
    brace_count = 0
    start_idx = -1
    for i, char in enumerate(response_text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                try:
                    return json.loads(response_text[start_idx:i+1])
                except json.JSONDecodeError:
                    start_idx = -1
                    brace_count = 0

    print(f"❌ 无法从响应中提取JSON: {response_text[:200]}...")
    return {}


# 测试用例
test_cases = [
    # 测试1: markdown包裹的JSON（用户日志中的格式）
    """```json
{
  "type": "line",
  "title": null,
  "page": 4,
  "axes": {
    "x": {
      "type": "category",
      "labels": ["15", "16", "17"]
    }
  }
}
```""",

    # 测试2: 无换行的markdown
    """```json{"type": "bar", "title": "test"}```""",

    # 测试3: 混合文本
    """Here is the result:
```json
{
  "type": "combo",
  "title": "EU risk premium"
}
```
Hope this helps!""",

    # 测试4: 纯JSON（无markdown）
    """{"type": "line", "title": "test"}""",

    # 测试5: 嵌套JSON
    """{
  "schema_version": "1.3.1",
  "doc": {
    "doc_id": "test",
    "title": "Test Doc"
  }
}"""
]

print("=" * 60)
print("JSON提取函数测试")
print("=" * 60)

for i, test in enumerate(test_cases, 1):
    print(f"\n【测试 {i}】")
    print(f"输入: {test[:80]}{'...' if len(test) > 80 else ''}")
    result = _extract_json_from_response(test)
    if result:
        print(f"✅ 成功提取: {list(result.keys())}")
    else:
        print(f"❌ 提取失败")

print("\n" + "=" * 60)
