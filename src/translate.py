# translate.py - Claude で翻訳
# 長い原稿は自動的にバッチ分割して翻訳

import json

from claude_api import call_json

# バッチ分割の設定
BATCH_MAX_LINES = 80      # 1バッチあたりの最大行数
CONTEXT_OVERLAP = 3        # バッチ間で共有するコンテキスト行数

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


def _build_preamble(context: str = "",
                    speaker_profiles: list[dict] | None = None,
                    speakers: dict | None = None) -> str:
    """翻訳リクエストの前文（コンテキスト + 話者情報）を構築する。"""
    parts = []

    if context:
        parts.append(f"コンテキスト: {context}")

    if speaker_profiles:
        profiles_text = "話者プロファイル:\n"
        for sp in speaker_profiles:
            profiles_text += (f"- {sp.get('id', '?')}: "
                              f"{sp.get('name', 'Unknown')} "
                              f"({sp.get('gender', '?')}, {sp.get('role', '?')})"
                              f" — {sp.get('description', '')}\n")
        parts.append(profiles_text)
    elif speakers:
        profiles_text = "話者プロファイル（音声分析による推定）:\n"
        for sp_id, features in speakers.items():
            gender = features.get("gender_hint", "unknown")
            profiles_text += f"- {sp_id}: ({gender})\n"
        parts.append(profiles_text)

    return "\n\n".join(parts)


def _build_user_message(preamble: str, transcript_lines: list[str],
                        prev_context: list[dict] | None = None) -> str:
    """翻訳リクエストのユーザーメッセージを構築する。"""
    user_message = ""

    if preamble:
        user_message += preamble + "\n\n"

    if prev_context:
        user_message += "（前のパートの末尾。口調の参考用、翻訳不要）:\n"
        for seg in prev_context:
            user_message += f"[{seg['speaker']}] {seg['text']}\n"
        user_message += "\n"

    transcript = "\n".join(transcript_lines)
    user_message += f"以下のテキストを日本語に翻訳してください:\n\n{transcript}"
    return user_message


def _call_and_parse(user_message: str) -> list[dict]:
    """Claude APIを呼び出してJSONをパースする。途中切れ時はValueErrorを送出。"""
    return call_json(
        model="claude-sonnet-4-20250514",
        max_tokens=16384,
        system=SYSTEM_PROMPT,
        user_message=user_message,
    )


def _translate_batch(preamble: str, transcript_lines: list[str],
                     prev_context: list[dict] | None = None) -> list[dict]:
    """1バッチ分の翻訳を実行する。途中切れ時はバッチを半分に分割してリトライ。"""
    user_message = _build_user_message(preamble, transcript_lines, prev_context)

    try:
        return _call_and_parse(user_message)
    except (json.JSONDecodeError, ValueError) as e:
        if len(transcript_lines) <= 10:
            raise  # これ以上分割できない

        mid = len(transcript_lines) // 2
        print(f"    ⚠ レスポンス途中切れ ({e})。バッチを分割してリトライ...")

        # 前半
        first_half = _translate_batch(preamble, transcript_lines[:mid], prev_context)
        # 後半（前半末尾をコンテキストに）
        ctx = first_half[-CONTEXT_OVERLAP:] if first_half else None
        second_half = _translate_batch(preamble, transcript_lines[mid:], ctx)

        return first_half + second_half


def translate(transcript: str, speakers: dict | None = None,
              context: str = "",
              speaker_profiles: list[dict] | None = None) -> list[dict]:
    """書き起こしテキストを日本語に翻訳する。

    長い原稿は自動的にバッチ分割して翻訳する。

    Args:
        transcript: 原文テキスト（話者ラベル付き）
        speakers: 話者の声質情報
        context: メタデータ分析から得たコンテキスト
        speaker_profiles: 話者プロファイル

    Returns:
        [{"speaker": "Speaker_1", "text": "..."}, ...]
    """
    preamble = _build_preamble(context, speaker_profiles, speakers)

    lines = [l for l in transcript.split("\n") if l.strip()]
    total_chars = len(transcript)

    # 短い原稿はそのまま翻訳
    if len(lines) <= BATCH_MAX_LINES:
        print(f"\nClaude 翻訳中... ({total_chars} 文字)")
        result = _translate_batch(preamble, lines)
        print(f"  翻訳完了: {len(result)} セグメント")
        return result

    # 長い原稿はバッチ分割
    batches = []
    for i in range(0, len(lines), BATCH_MAX_LINES):
        batches.append(lines[i:i + BATCH_MAX_LINES])

    print(f"\nClaude 翻訳中... ({total_chars} 文字, {len(batches)} バッチ)")

    all_results = []
    prev_context = None

    for i, batch in enumerate(batches):
        print(f"  バッチ {i + 1}/{len(batches)} ({len(batch)} 行)...")
        result = _translate_batch(preamble, batch, prev_context)
        all_results.extend(result)
        print(f"    → {len(result)} セグメント")

        # 次のバッチ用のコンテキスト
        prev_context = result[-CONTEXT_OVERLAP:] if result else None

    print(f"  翻訳完了: 合計 {len(all_results)} セグメント")
    return all_results
