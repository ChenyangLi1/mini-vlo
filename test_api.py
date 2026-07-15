"""Minimal DashScope API key and network connectivity test."""

from __future__ import annotations

import os
import traceback
from pathlib import Path

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)


ROOT_DIR = Path(__file__).resolve().parent
ENV_PATH = ROOT_DIR / ".env"
MODEL = "qwen3-vl-flash"


def mask_api_key(api_key: str) -> str:
    """Show enough of the key to identify it without exposing the secret."""
    if len(api_key) <= 8:
        return f"{api_key[:2]}***{api_key[-2:]}"
    return f"{api_key[:4]}{'*' * min(len(api_key) - 8, 20)}{api_key[-4:]}"


def print_api_error(error: Exception) -> None:
    """Print common OpenAI-compatible API error details."""
    print(f"错误类型: {type(error).__name__}")
    print(f"错误信息: {error}")

    status_code = getattr(error, "status_code", None)
    request_id = getattr(error, "request_id", None)
    body = getattr(error, "body", None)

    if status_code is not None:
        print(f"HTTP 状态码: {status_code}")
    if request_id:
        print(f"Request ID: {request_id}")
    if body is not None:
        print(f"API 错误响应: {body}")


def main() -> int:
    print("=== DashScope API 连通性测试 ===")
    print(f".env 路径: {ENV_PATH}")

    if not ENV_PATH.is_file():
        print("失败：项目根目录下没有找到 .env 文件。")
        print("请复制 .env.example 为 .env，并填写有效的 DASHSCOPE_API_KEY。")
        return 1

    loaded = load_dotenv(ENV_PATH, override=False)
    print(f".env 加载状态: {'成功' if loaded else '未加载到新变量'}")

    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("API Key 读取状态: 失败")
        print("请在 .env 中设置 DASHSCOPE_API_KEY。")
        return 1

    print(f"API Key 读取状态: 成功 ({mask_api_key(api_key)})")

    base_url = os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    timeout = float(os.getenv("VLM_TIMEOUT", "60"))

    print(f"Base URL: {base_url}")
    print(f"测试模型: {MODEL}")

    try:
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": "Hello, are you working? Reply briefly.",
                }
            ],
            temperature=0,
            max_tokens=32,
        )

        content = response.choices[0].message.content or ""
        print("\nAPI 调用状态: 成功")
        print(f"模型回复: {content}")
        if getattr(response, "_request_id", None):
            print(f"Request ID: {response._request_id}")
        return 0

    except AuthenticationError as error:
        print("\nAPI 调用状态: 失败（认证失败）")
        print_api_error(error)
    except APIConnectionError as error:
        print("\nAPI 调用状态: 失败（网络连接错误）")
        print_api_error(error)
        print(f"底层原因: {error.__cause__!r}")
    except RateLimitError as error:
        print("\nAPI 调用状态: 失败（限流或额度不足）")
        print_api_error(error)
    except APIStatusError as error:
        print("\nAPI 调用状态: 失败（API 返回错误状态）")
        print_api_error(error)
    except OpenAIError as error:
        print("\nAPI 调用状态: 失败（OpenAI 客户端错误）")
        print_api_error(error)
    except Exception as error:
        print("\nAPI 调用状态: 失败（未预期错误）")
        print_api_error(error)
        traceback.print_exc()

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
