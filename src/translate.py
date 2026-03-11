# translate.py - Claude で翻訳

import os
import json
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

SYSTEM_PROMPT = """\
あなたはプロの翻訳者です。外国語のポッドキャスト/動画の書き起こしテキストを日本語に翻訳してください。

## 入力形式
テキストは話者ラベル付きで提供されます:
[Speaker_1] Hello, welcome to the show.
[Speaker_2] Thanks for having me.

## ルール
- 原文に忠実に翻訳する（意訳・再構成はしない）
- 冗長な部分（繰り返し、フィラー、脱線）は自然にカットしてよい
- 話者ラベル（Speaker_1, Speaker_2 等）はそのまま維持する。勝手に名前を変えない
- 固有名詞（人名、地名、大会名、ブランド名）は原語のまま残す
- 自然で聞きやすい日本語にする（書き言葉ではなく話し言葉）
- 話者の交代を正確に反映する。同じ話者の連続発言は1つにまとめてよい

## 口調の重要なルール
- 話者プロファイル（性別・役割）が提供される場合、それに合った日本語の口調にすること
- 男性話者: 「〜だよね」「〜だと思う」「〜じゃないかな」など男性的な話し言葉
- 女性話者: 「〜よね」「〜だと思うわ」「〜じゃないかしら」など女性的な話し言葉
- ただし現代的で自然な口調にする。過度にステレオタイプな言い方は避ける
- 役割（host/guest）に応じたトーンも反映する

## 出力形式
JSON配列で出力してください。各要素は以下の形式:
[
  {"speaker": "Speaker_1", "text": "日本語翻訳テキスト"},
  {"speaker": "Speaker_2", "text": "日本語翻訳テキスト"},
  ...
]

1つの要素のテキストは200文字以内を目安にしてください（長い場合は同じ話者のまま分割）。

JSONのみ出力し、他の説明は不要です。
"""


def translate(transcript: str, speakers: dict | None = None,
              context: str = "",
              speaker_profiles: list[dict] | None = None) -> list[dict]:
    """書き起こしテキストを日本語に翻訳する。

    Args:
        transcript: 原文テキスト（話者ラベル付き）
        speakers: 話者の声質情報
        context: メタデータ分析から得たコンテキスト
        speaker_profiles: 話者プロファイル [{"id", "name", "role", "gender"}, ...]

    Returns:
        [{"speaker": "Speaker_1", "text": "..."}, ...]
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY が .env に設定されていません")

    client = anthropic.Anthropic(api_key=api_key)

    user_message = ""

    if context:
        user_message += f"コンテキスト: {context}\n\n"

    if speaker_profiles:
        profiles_text = "話者プロファイル:\n"
        for sp in speaker_profiles:
            profiles_text += (f"- {sp.get('id', '?')}: "
                              f"{sp.get('name', 'Unknown')} "
                              f"({sp.get('gender', '?')}, {sp.get('role', '?')})"
                              f" — {sp.get('description', '')}\n")
        user_message += profiles_text + "\n"
    elif speakers:
        # メタデータがなくても voice_features の gender_hint から性別を伝える
        profiles_text = "話者プロファイル（音声分析による推定）:\n"
        for sp_id, features in speakers.items():
            gender = features.get("gender_hint", "unknown")
            profiles_text += f"- {sp_id}: ({gender})\n"
        user_message += profiles_text + "\n"

    user_message += f"以下のテキストを日本語に翻訳してください:\n\n{transcript}"

    print(f"\nClaude 翻訳中... ({len(transcript)} 文字)")

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    text = response.content[0].text.strip()

    # JSON 抽出（コードブロックで囲まれている場合にも対応）
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])

    result = json.loads(text)
    print(f"  翻訳完了: {len(result)} セグメント")
    return result
