#!/usr/bin/env python3
"""快速诊断脚本：验证 Minimax API 是否可连通"""

import os
import sys
from pathlib import Path

# 加载 .env
sys.path.insert(0, str(Path(__file__).parent / "src"))
from support_copilot.config import Config

print("=" * 60)
print("Minimax 配置诊断")
print("=" * 60)

print(f"\n1. ANTHROPIC_BASE_URL: {Config.ANTHROPIC_BASE_URL or '❌ 未设置'}")
print(f"2. ANTHROPIC_API_KEY: {'✅ 已设置' if Config.ANTHROPIC_API_KEY else '❌ 未设置'}")
print(f"3. LLM_MODEL: {Config.LLM_MODEL}")
print(f"4. MAX_TOKENS: {Config.MAX_TOKENS}")

if not Config.ANTHROPIC_API_KEY:
    print("\n⚠️  未检测到 ANTHROPIC_API_KEY，将使用离线抽取式回答")
    print("请在 .env 文件中设置：")
    print("  ANTHROPIC_API_KEY=你的key")
    sys.exit(1)

print("\n5. 尝试初始化 LLM answerer...")
from support_copilot.llm import build_minimax_chat_answerer_from_env

answerer = build_minimax_chat_answerer_from_env()
if answerer:
    print("   ✅ LLM answerer 初始化成功")
else:
    print("   ❌ LLM answerer 初始化失败")
    sys.exit(1)

print("\n6. 测试真实 API 调用（会产生费用）...")
try:
    result = answerer(
        "什么是spot实例?",
        "Spot实例是一种折扣的按需付费实例，但可能随时被回收。"
    )
    print(f"   ✅ API 调用成功")
    print(f"\n回答内容：\n{result}\n")
except Exception as e:
    print(f"   ❌ API 调用失败: {e}")
    sys.exit(1)

print("=" * 60)
print("✅ Minimax 集成正常，可以使用")
print("=" * 60)
