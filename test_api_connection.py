#!/usr/bin/env python3
"""
测试OpenRouter API连接和超时配置
"""

import os
import asyncio
import httpx
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
                        print(f"  - 加载: {key}={value[:20]}..." if len(value) > 20 else f"  - 加载: {key}={value}")
        return True
    except Exception as e:
        print(f"❌ 加载.env文件失败: {e}")
        return False


async def test_api_connection():
    """测试API连接和超时配置"""

    print("\n" + "=" * 60)
    print("步骤1: 加载环境变量")
    print("=" * 60)

    # 加载.env文件
    if not load_env_file():
        return False

    # 读取API密钥
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        print("\n❌ 错误: OPENROUTER_API_KEY环境变量未设置或为空")
        print("请检查.env文件中的OPENROUTER_API_KEY配置")
        return False

    print(f"\n✓ API密钥已加载: {api_key[:20]}...{api_key[-10:]}")

    print("\n" + "=" * 60)
    print("步骤2: 配置超时参数")
    print("=" * 60)

    # 配置超时 - 与batch_pdf_processor.py中相同的配置
    timeout_config = httpx.Timeout(
        connect=60.0,  # 连接超时: 60秒
        read=600.0,    # 读取超时: 10分钟
        write=60.0,    # 写入超时: 60秒
        pool=60.0      # 连接池超时: 60秒
    )

    print(f"✓ 超时配置: connect={timeout_config.connect}s, read={timeout_config.read}s")

    print("\n" + "=" * 60)
    print("步骤3: 创建AsyncOpenAI客户端")
    print("=" * 60)

    # 创建客户端
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        timeout=timeout_config,
        max_retries=3
    )

    print("✓ AsyncOpenAI客户端创建成功")

    print("\n" + "=" * 60)
    print("步骤4: 测试API调用")
    print("=" * 60)

    try:
        print("\n正在调用OpenRouter API...")
        print(f"  - 模型: google/gemini-2.5-flash")
        print(f"  - Max tokens: 50")
        print(f"  - Temperature: 0.0")
        print(f"  - Response format: JSON\n")

        # 简单的测试请求
        response = await client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=[
                {"role": "user", "content": "Say 'Hello, API test successful!' in JSON format with a 'message' field."}
            ],
            max_tokens=50,
            temperature=0.0,
            response_format={"type": "json_object"}
        )

        print("\n✅ API调用成功!")
        print(f"响应内容: {response.choices[0].message.content}")
        print(f"模型: {response.model}")
        print(f"Token使用: {response.usage.total_tokens}")

        await client.close()
        return True

    except httpx.ConnectTimeout as e:
        print(f"\n❌ 连接超时错误: {e}")
        print("可能的原因:")
        print("  1. 网络连接不稳定")
        print("  2. 需要配置代理")
        print("  3. 防火墙阻止连接")
        await client.close()
        return False

    except httpx.ReadTimeout as e:
        print(f"\n❌ 读取超时错误: {e}")
        print("API响应时间过长")
        await client.close()
        return False

    except Exception as e:
        print(f"\n❌ 错误: {type(e).__name__}: {e}")
        await client.close()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("OpenRouter API 连接测试")
    print("=" * 60)

    success = asyncio.run(test_api_connection())

    print("\n" + "=" * 60)
    print("测试结果")
    print("=" * 60)
    if success:
        print("✅ 测试通过! API连接正常，可以继续批量处理。")
    else:
        print("❌ 测试失败! 请检查网络连接和API配置。")
    print("=" * 60)
