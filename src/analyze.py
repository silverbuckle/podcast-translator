# analyze.py - メタデータから出演者情報を推定する

import os
import json
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

SYSTEM_PROMPT = """\
あなたはポッドキャスト/動画のメタデータを分析し、出演者情報を推定するアシスタントです。

与えられたメタデータ（タイトル、説明文、チャンネル名、字幕テキスト等）から、
以下の情報をJSON形式で出力してください。

## 出力形式
```json
{
  "num_speakers": 2,
  "speakers": [
    {
      "id": "Speaker_1",
      "name": "John Smith",
      "role": "host",
      "gender": "male",
      "description": "番組のメインホスト"
    },
    {
      "id": "Speaker_2",
      "name": "Jane Doe",
      "role": "guest",
      "gender": "female",
      "description": "今回のゲスト、○○の専門家"
    }
  ],
  "context": "テクノロジーに関するインタビュー番組",
  "confidence": "high"
}
```

## ルール
- num_speakers: 音声に登場する話者の数を推定（ナレーションのみなら1）
- name: 特定できない場合は "Unknown" とする
- role: "host", "guest", "co-host", "narrator", "interviewer", "interviewee" のいずれか
- gender: "male", "female", "unknown" のいずれか
- confidence: 推定の確信度 "high", "medium", "low"
- 情報が少ない場合でも、タイトルや形式から最善の推測をする
  - "Interview with X" → 2人（host + guest）
  - "X and Y discuss..." → 2人
  - "X reacts to..." → 1人
  - Podcast は通常 2-3人（host + co-host or guest）
- 字幕テキストから話者の切り替わりパターンも手がかりにする

JSONのみ出力し、他の説明は不要です。
"""


def analyze_metadata(metadata: dict) -> dict:
    """メタデータから出演者情報を推定する。

    Args:
        metadata: download() が返すメタデータ

    Returns:
        {
            "num_speakers": int,
            "speakers": [{"id", "name", "role", "gender", "description"}, ...],
            "context": str,
            "confidence": str,
        }
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return _fallback()

    # 分析に使う情報を構築
    parts = []

    if metadata.get("title"):
        parts.append(f"タイトル: {metadata['title']}")
    if metadata.get("channel"):
        parts.append(f"チャンネル: {metadata['channel']}")
    if metadata.get("show_title"):
        parts.append(f"番組名: {metadata['show_title']}")
    if metadata.get("show_author"):
        parts.append(f"番組著者: {metadata['show_author']}")
    if metadata.get("description"):
        parts.append(f"説明文:\n{metadata['description'][:1500]}")
    if metadata.get("show_description"):
        parts.append(f"番組説明:\n{metadata['show_description']}")
    if metadata.get("tags"):
        parts.append(f"タグ: {', '.join(metadata['tags'][:20])}")
    if metadata.get("captions"):
        parts.append(f"字幕テキスト（冒頭）:\n{metadata['captions'][:1000]}")

    if not parts:
        return _fallback()

    user_message = "\n\n".join(parts)
    print(f"\n  メタデータ分析中 (Claude)...")

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        text = response.content[0].text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])

        result = json.loads(text)

        # 結果を表示
        print(f"  推定話者数: {result.get('num_speakers', '?')}"
              f" (確信度: {result.get('confidence', '?')})")
        for sp in result.get("speakers", []):
            print(f"    {sp.get('id')}: {sp.get('name', '?')} "
                  f"({sp.get('role', '?')}, {sp.get('gender', '?')})")
        if result.get("context"):
            print(f"  コンテキスト: {result['context']}")

        return result

    except Exception as e:
        print(f"  メタデータ分析失敗: {e}")
        return _fallback()


def _fallback() -> dict:
    """メタデータ分析が失敗した場合のフォールバック。"""
    return {
        "num_speakers": None,
        "speakers": [],
        "context": "",
        "confidence": "low",
    }
