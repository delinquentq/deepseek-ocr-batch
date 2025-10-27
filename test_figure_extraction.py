#!/usr/bin/env python3
"""
测试图表提取功能 - 诊断阶段B卡住问题
"""

import os
import asyncio
import httpx
import base64
import json
from pathlib import Path
from openai import AsyncOpenAI


def load_env_file():
    """加载.env文件到环境变量"""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        print(f"⚠️ 警告: .env文件不存在: {env_path}")
        return False

    print(f"✓ 找到.env文件: {env_path}")

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and value:
                        os.environ[key] = value
        print(f"✓ .env文件加载成功")
        return True
    except Exception as e:
        print(f"❌ 加载.env文件失败: {e}")
        return False


async def test_figure_extraction():
    """测试单个图表的识别"""

    print("\n" + "=" * 60)
    print("步骤1: 加载环境变量")
    print("=" * 60)

    # 加载.env文件
    if not load_env_file():
        return False

    # 加载API密钥
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        print("\n❌ 错误: OPENROUTER_API_KEY环境变量未设置或为空")
        print("请检查.env文件中的OPENROUTER_API_KEY配置")
        return False

    print(f"✓ API密钥已加载: {api_key[:20]}...{api_key[-10:]}")

    print("\n" + "=" * 60)
    print("步骤2: 查找测试图像")
    print("=" * 60)

    # 查找测试图像
    test_image_dir = Path("/home/qxx/DeepSeek-OCR/DeepSeek-OCR-master/deepseek-ocr-batch/output_results/2025-09-28/Aerospace Defense Space industry revolution Who wins Or Is it wrong to wish on space hardwareAerospace Defense Space ind/images")

    if not test_image_dir.exists():
        print(f"❌ 测试图像目录不存在: {test_image_dir}")
        return False

    image_files = list(test_image_dir.glob("*.jpg"))
    if not image_files:
        print(f"❌ 没有找到图像文件")
        return False

    test_image = image_files[0]
    print(f"✓ 使用测试图像: {test_image}")

    # 编码图像为Base64
    try:
        with open(test_image, "rb") as f:
            image_data = f.read()
        b64_image = base64.b64encode(image_data).decode('utf-8')
        print(f"✓ 图像已编码 (大小: {len(image_data)} bytes, Base64: {len(b64_image)} chars)")
    except Exception as e:
        print(f"❌ 图像编码失败: {e}")
        return False

    # 配置超时
    timeout_config = httpx.Timeout(
        connect=60.0,  # 连接超时: 60秒
        read=600.0,    # 读取超时: 10分钟
        write=60.0,    # 写入超时: 60秒
        pool=60.0      # 连接池超时: 60秒
    )

    print(f"\n✓ 超时配置: connect={timeout_config.connect}s, read={timeout_config.read}s")

    print("\n" + "=" * 60)
    print("步骤4: 创建AsyncOpenAI客户端")
    print("=" * 60)

    # 创建AsyncOpenAI客户端
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        timeout=timeout_config,
        max_retries=3
    )

    print("✓ AsyncOpenAI客户端创建成功")

    print("\n" + "=" * 60)
    print("步骤5: 调用API识别图表")
    print("=" * 60)

    # 构建请求消息
    messages = [
        {
            "role": "system",
            "content": "你是专业的图表数据提取专家。请识别图表类型（柱状图/折线图/饼图/表格等），并提取其中的所有数据。"
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_image}"
                    }
                },
                {
                    "type": "text",
                    "text": """请分析这张图表，提取以下信息并以JSON格式输出：

1. 图表类型（type）：bar/line/pie/area/scatter/heatmap/waterfall/combo/other
2. 图表标题（title）
3. 坐标轴信息（axes）：
   - x轴：类型、标签
   - y轴：单位、范围
4. 数据系列（series）：每个系列包含name、unit、values数组

输出格式示例：
{
  "type": "bar",
  "title": "Revenue Growth",
  "page": 1,
  "axes": {
    "x": {"type": "category", "labels": ["Q1", "Q2", "Q3", "Q4"]},
    "y": {"unit": "USD million", "range": {"min": 0, "max": 100}}
  },
  "series": [
    {"name": "Revenue", "unit": "USD million", "values": [20, 30, 45, 60]}
  ]
}

仅输出JSON，不要其他文字。"""
                }
            ]
        }
    ]

    try:
        print("\n正在调用OpenRouter API...")
        print(f"  - 模型: google/gemini-2.5-flash")
        print(f"  - Max tokens: 1536")
        print(f"  - Temperature: 0.0")
        print(f"  - 图像: {test_image.name}\n")

        response = await client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=messages,
            max_tokens=1536,
            temperature=0.0
        )

        print(f"✅ API调用成功!")
        print(f"\n响应内容:")
        print("=" * 60)
        print(response.choices[0].message.content)
        print("=" * 60)
        print(f"\n模型: {response.model}")
        print(f"Token使用: {response.usage.total_tokens} (输入: {response.usage.prompt_tokens}, 输出: {response.usage.completion_tokens})")

        # 尝试解析JSON
        try:
            result_json = json.loads(response.choices[0].message.content)
            print("\n✓ JSON解析成功")
            print(json.dumps(result_json, indent=2, ensure_ascii=False))
        except json.JSONDecodeError as e:
            print(f"\n⚠️ JSON解析失败: {e}")

        await client.close()
        return True

    except httpx.ConnectTimeout as e:
        print(f"\n❌ 连接超时错误: {e}")
        print("\n可能的原因:")
        print("  1. 网络连接不稳定")
        print("  2. 需要配置HTTP代理")
        print("  3. 防火墙阻止连接到openrouter.ai")
        print("  4. DNS解析问题")
        print("\n诊断建议:")
        print("  - 测试网络连接: ping openrouter.ai")
        print("  - 测试HTTP连接: curl -I https://openrouter.ai/api/v1")
        print("  - 检查代理设置: echo $HTTP_PROXY $HTTPS_PROXY")
        await client.close()
        return False

    except httpx.ReadTimeout as e:
        print(f"\n❌ 读取超时错误: {e}")
        print("API响应时间过长（超过600秒）")
        await client.close()
        return False

    except Exception as e:
        print(f"\n❌ 未知错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        await client.close()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("图表提取功能测试")
    print("=" * 60)

    success = asyncio.run(test_figure_extraction())

    print("\n" + "=" * 60)
    print("测试结果")
    print("=" * 60)
    if success:
        print("✅ 测试通过! 图表识别API调用正常。")
        print("\n建议: 检查batch_pdf_processor.py中的异常处理逻辑")
    else:
        print("❌ 测试失败! 这就是阶段B卡住的原因。")
        print("\n解决方案:")
        print("  1. 如果是连接超时: 配置HTTP代理或检查网络连接")
        print("  2. 如果是读取超时: API调用时间过长，可能需要减少图像大小")
        print("  3. 检查防火墙设置是否允许访问openrouter.ai")
    print("=" * 60)
