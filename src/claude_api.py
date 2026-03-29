# claude_api.py - Claude API 共通ヘルパー
#
# 全モジュール共通の API 呼び出し・JSON パース・エラーハンドリングを一元管理。
# ストリーミング接続で長時間リクエストにも対応。

import json
import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    """認証済み Anthropic クライアントを返す。キー未設定時は ValueError。"""
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY が .env に設定されていません")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def has_api_key() -> bool:
    """ANTHROPIC_API_KEY が設定されているか。"""
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def _strip_markdown_fence(text: str) -> str:
    """```json ... ``` のフェンスを除去する。"""
    if text.startswith("```"):
        lines = text.split("\n")
        return "\n".join(lines[1:-1])
    return text


def call_json(*, model: str, max_tokens: int,
              user_message: str, system: str | None = None) -> dict | list:
    """Claude API をストリーミングで呼び出し、JSON をパースして返す。

    Args:
        model: モデル名 (e.g. "claude-sonnet-4-20250514")
        max_tokens: 最大トークン数
        user_message: ユーザーメッセージ
        system: システムプロンプト（省略可）

    Returns:
        パース済み JSON (dict or list)

    Raises:
        ValueError: レスポンス途中切れ (stop_reason != "end_turn")
        json.JSONDecodeError: JSON パース失敗
    """
    client = get_client()

    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": user_message}],
    )
    if system:
        kwargs["system"] = system

    with client.messages.stream(**kwargs) as stream:
        response = stream.get_final_message()

    if response.stop_reason != "end_turn":
        raise ValueError(
            f"レスポンス途中切れ (stop_reason={response.stop_reason})")

    text = _strip_markdown_fence(response.content[0].text.strip())
    return json.loads(text)
