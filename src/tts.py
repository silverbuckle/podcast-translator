# tts.py - Gemini Flash TTS で音声生成
# trail-podcast の tts.py をベースに汎用化

import os
import sys
import io
import json
import re
import time
from pathlib import Path

# Python 3.13+ audioop 互換
import audioop as _audioop
sys.modules.setdefault("pyaudioop", _audioop)

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydub import AudioSegment

load_dotenv(Path(__file__).parent.parent / ".env")

# ------------------------------------------------------------------ 設定 --

AUDIO_DIR = Path(__file__).parent.parent / "output" / "audio"
CUSTOM_READINGS_PATH = Path(__file__).parent.parent / "input" / "custom_readings.json"

MODEL = "gemini-2.5-flash-preview-tts"
READINGS_MODEL = "gemini-2.5-flash"

# チャンクサイズ上限
CHUNK_MAX_CHARS = 3000

# API呼び出し間の待機時間
API_CALL_INTERVAL = 1.0

# Gemini TTS 全30プリセットボイス
# gender: male/female, tone: 声の特徴, energy: 声のエネルギー感
VOICE_CATALOG = {
    # --- 男性ボイス (15種) ---
    "Puck":           {"gender": "male",   "tone": "upbeat",        "energy": "high"},
    "Charon":         {"gender": "male",   "tone": "informative",   "energy": "mid"},
    "Fenrir":         {"gender": "male",   "tone": "excitable",     "energy": "high"},
    "Orus":           {"gender": "male",   "tone": "firm",          "energy": "mid"},
    "Enceladus":      {"gender": "male",   "tone": "breathy",       "energy": "low"},
    "Iapetus":        {"gender": "male",   "tone": "clear",         "energy": "mid"},
    "Umbriel":        {"gender": "male",   "tone": "easy-going",    "energy": "low"},
    "Algieba":        {"gender": "male",   "tone": "smooth",        "energy": "low"},
    "Algenib":        {"gender": "male",   "tone": "gravelly",      "energy": "mid"},
    "Rasalgethi":     {"gender": "male",   "tone": "informative",   "energy": "mid"},
    "Alnilam":        {"gender": "male",   "tone": "firm",          "energy": "high"},
    "Schedar":        {"gender": "male",   "tone": "even",          "energy": "mid"},
    "Pulcherrima":    {"gender": "male",   "tone": "forward",       "energy": "high"},
    "Achird":         {"gender": "male",   "tone": "friendly",      "energy": "mid"},
    "Sadachbia":      {"gender": "male",   "tone": "lively",        "energy": "high"},
    "Sadaltager":     {"gender": "male",   "tone": "knowledgeable", "energy": "mid"},
    "Zubenelgenubi":  {"gender": "male",   "tone": "casual",        "energy": "low"},
    # --- 女性ボイス (13種) ---
    "Zephyr":         {"gender": "female", "tone": "bright",        "energy": "high"},
    "Kore":           {"gender": "female", "tone": "firm",          "energy": "mid"},
    "Leda":           {"gender": "female", "tone": "youthful",      "energy": "high"},
    "Aoede":          {"gender": "female", "tone": "breezy",        "energy": "mid"},
    "Callirrhoe":     {"gender": "female", "tone": "easy-going",    "energy": "low"},
    "Autonoe":        {"gender": "female", "tone": "bright",        "energy": "high"},
    "Despina":        {"gender": "female", "tone": "smooth",        "energy": "low"},
    "Erinome":        {"gender": "female", "tone": "clear",         "energy": "mid"},
    "Laomedeia":      {"gender": "female", "tone": "upbeat",        "energy": "high"},
    "Achernar":       {"gender": "female", "tone": "soft",          "energy": "low"},
    "Gacrux":         {"gender": "female", "tone": "mature",        "energy": "mid"},
    "Vindemiatrix":   {"gender": "female", "tone": "gentle",        "energy": "low"},
    "Sulafat":        {"gender": "female", "tone": "warm",          "energy": "mid"},
}

# デフォルトボイス
DEFAULT_VOICE = "Charon"


# -------------------------------------------------------- 読み辞書 --

def _load_custom_readings() -> dict[str, str]:
    if not CUSTOM_READINGS_PATH.exists():
        return {}
    with open(CUSTOM_READINGS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _fix_readings(text: str, readings: dict[str, str]) -> str:
    for word, reading in sorted(readings.items(), key=lambda x: -len(x[0])):
        text = text.replace(word, reading)
    return text


def _extract_new_terms(segments: list[dict], existing: dict[str, str]) -> list[str]:
    """翻訳済みテキストから辞書に未登録の固有名詞を抽出する。"""
    candidates: set[str] = set()

    for seg in segments:
        text = seg.get("text", "")
        if not text:
            continue
        # 英語の複合名
        for m in re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', text):
            candidates.add(m)
        # 英語の単独固有名詞（4文字以上）
        for m in re.findall(r'[A-Z][a-z]{3,}', text):
            candidates.add(m)
        # 英字略語
        for m in re.findall(r'[A-Z]{2,}', text):
            candidates.add(m)

    candidates = {t for t in candidates if len(t) >= 2}
    new_terms = [t for t in sorted(candidates) if t not in existing
                 and not any(t in k or k in t for k in existing)]
    return new_terms


def _update_readings_with_llm(client: genai.Client, new_terms: list[str],
                              existing: dict[str, str]) -> dict[str, str]:
    if not new_terms:
        return existing

    existing_examples = "\n".join(f"  {k} → {v}" for k, v in list(existing.items())[:10])
    prompt = f"""以下はポッドキャスト台本のTTS（音声合成）用の読み辞書です。

既存の辞書例:
{existing_examples}

以下の語句について、日本語TTSで正しく読まれるよう、カタカナまたはひらがなの読みを生成してください。
- 英語の人名・地名 → カタカナ
- 英語の略語 → 一文字ずつカタカナ（例: GPS → ジーピーエス）
- 日本語の固有名詞 → ひらがな

必ず以下のJSON形式のみ出力してください（説明不要）:
{{"語句": "読み", ...}}

語句リスト:
{json.dumps(new_terms, ensure_ascii=False, indent=2)}"""

    try:
        response = client.models.generate_content(model=READINGS_MODEL, contents=prompt)
        text = response.text.strip()
        match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
        if match:
            new_readings = json.loads(match.group())
            return {**existing, **new_readings}
    except Exception as e:
        print(f"  警告: 読み自動生成に失敗: {e}")
    return existing


def _save_custom_readings(readings: dict[str, str]) -> None:
    CUSTOM_READINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {"_comment": "TTS 読み辞書。固有名詞を正しい読みに置換する。"}
    data.update(readings)
    with open(CUSTOM_READINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def update_custom_readings(client: genai.Client, segments: list[dict]) -> dict[str, str]:
    existing = _load_custom_readings()
    new_terms = _extract_new_terms(segments, existing)
    if not new_terms:
        print(f"  辞書更新: 新規用語なし（既存 {len(existing)} 件）")
        return existing

    print(f"  辞書更新: 未登録用語 {len(new_terms)} 件を検出")
    updated = _update_readings_with_llm(client, new_terms, existing)
    added = len(updated) - len(existing)
    if added > 0:
        _save_custom_readings(updated)
        print(f"  辞書更新: {added} 件追加（合計 {len(updated)} 件）")
    return updated


# -------------------------------------------------------------- helpers --

def _pcm_to_audio_segment(pcm_data: bytes) -> AudioSegment:
    return AudioSegment(data=pcm_data, sample_width=2, frame_rate=24000, channels=1)


def _assign_voices(segments: list[dict],
                   voice_features: dict[str, dict] | None = None) -> dict[str, str]:
    """話者ごとに Gemini TTS ボイスを声質分析結果に基づいて割り当てる。

    F0（基本周波数）とエネルギーから、30種のプリセットボイスの中から
    最も近いキャラクターを選択する。
    """
    speakers = sorted(set(seg["speaker"] for seg in segments))

    if not voice_features:
        voice_pool = list(VOICE_CATALOG.keys())
        return {s: voice_pool[i % len(voice_pool)] for i, s in enumerate(speakers)}

    # F0 → エネルギー感のマッピング
    # 低い F0 (< 130Hz) → 落ち着いた低音 → breathy/smooth/easy-going
    # 中 F0 (130-180Hz) → 標準的な男性 → informative/clear/firm
    # やや高 F0 (180-230Hz) → 高めの男性 or 低めの女性 → warm/mature
    # 高 F0 (> 230Hz) → 女性 → bright/clear/breezy

    # エネルギー → ボイスのエネルギー感
    ENERGY_MAP = {"high": "high", "moderate": "mid", "low": "low"}

    # F0 帯域ごとの推奨ボイス（優先順）
    VOICE_PREFERENCE = {
        "male_low": ["Enceladus", "Algieba", "Umbriel", "Algenib",
                      "Zubenelgenubi", "Schedar"],
        "male_mid": ["Charon", "Iapetus", "Rasalgethi", "Orus",
                      "Achird", "Sadaltager"],
        "male_high": ["Puck", "Fenrir", "Sadachbia", "Alnilam",
                       "Pulcherrima"],
        "female_low": ["Achernar", "Despina", "Vindemiatrix",
                        "Callirrhoe", "Gacrux", "Sulafat"],
        "female_mid": ["Aoede", "Erinome", "Kore", "Sulafat", "Gacrux"],
        "female_high": ["Zephyr", "Leda", "Autonoe", "Laomedeia"],
    }

    mapping = {}
    used_voices: set[str] = set()

    for speaker in speakers:
        features = voice_features.get(speaker, {})
        gender = features.get("gender_hint", "unknown")
        f0 = features.get("estimated_f0_hz", 170)
        energy = features.get("energy", "moderate")

        # F0 とジェンダーから推奨リストを決定
        if gender == "male":
            if f0 < 130:
                pref_key = "male_low"
            elif f0 < 170:
                pref_key = "male_mid"
            else:
                pref_key = "male_high"
        elif gender == "female":
            if f0 < 220:
                pref_key = "female_low"
            elif f0 < 270:
                pref_key = "female_mid"
            else:
                pref_key = "female_high"
        else:
            pref_key = "male_mid" if f0 < 200 else "female_mid"

        # 推奨リストから未使用のボイスを選択
        preferred = VOICE_PREFERENCE.get(pref_key, [])
        target_energy = ENERGY_MAP.get(energy, "mid")

        best_voice = None
        best_score = -1

        for voice_name in preferred:
            if voice_name in used_voices:
                continue
            voice_info = VOICE_CATALOG.get(voice_name, {})
            score = 10  # 推奨リストにいるだけで基本スコア
            # エネルギー感がマッチすればボーナス
            if voice_info.get("energy") == target_energy:
                score += 5
            if score > best_score:
                best_score = score
                best_voice = voice_name

        # 推奨リストが全て使用済みの場合、同性別の残りから選ぶ
        if best_voice is None:
            for voice_name, voice_info in VOICE_CATALOG.items():
                if voice_name in used_voices:
                    continue
                if gender != "unknown" and voice_info["gender"] != gender:
                    continue
                best_voice = voice_name
                break

        if best_voice is None:
            best_voice = DEFAULT_VOICE

        mapping[speaker] = best_voice
        used_voices.add(best_voice)

    return mapping


def _build_speech_direction(voice_mapping: dict[str, str],
                            voice_features: dict[str, dict] | None = None) -> str:
    """話者ごとの詳細なスピーチディレクションを生成する。

    声の安定性を高めるため、各話者の声質を具体的に指示する。
    """
    # ボイスの特徴 → 詳細な音声指示のマッピング
    VOICE_DIRECTION = {
        # 男性
        "Charon":     "Male. Calm, professional, mid-pitched voice. Speaks at a steady, measured pace like a news anchor. Authoritative but warm.",
        "Enceladus":  "Male. Soft, breathy, low-pitched voice. Speaks slowly and gently, like a late-night radio host. Very calm and intimate.",
        "Puck":       "Male. Upbeat, lively, mid-to-high pitched voice. Speaks with energy and enthusiasm. Bright and engaging.",
        "Fenrir":     "Male. Excitable, passionate voice. Speaks with intensity and drive. High energy throughout.",
        "Orus":       "Male. Firm, authoritative, mid-pitched voice. Speaks with confidence and clarity. Strong and direct.",
        "Iapetus":    "Male. Clear, clean, mid-pitched voice. Speaks with precision and articulation. Professional narrator style.",
        "Umbriel":    "Male. Easy-going, relaxed, low-pitched voice. Speaks casually and unhurried. Laid-back tone.",
        "Algieba":    "Male. Smooth, flowing, low-pitched voice. Speaks with elegance. Rich and mellow.",
        "Algenib":    "Male. Gravelly, textured, low-to-mid pitched voice. Speaks with weight and presence. Distinctive and rugged.",
        "Rasalgethi": "Male. Informative, professional, mid-pitched voice. Speaks like a documentary narrator. Clear and knowledgeable.",
        "Alnilam":    "Male. Firm, confident, mid-to-high pitched voice. Speaks with conviction. Bold and assertive.",
        "Schedar":    "Male. Even, steady, mid-pitched voice. Speaks with consistency and balance. Reliable and measured.",
        "Pulcherrima":"Male. Forward, enterprising, mid-to-high pitched voice. Speaks with drive and purpose. Proactive tone.",
        "Achird":     "Male. Friendly, kind, mid-pitched voice. Speaks warmly and approachably. Gentle and personable.",
        "Sadachbia":  "Male. Lively, vivid, mid-to-high pitched voice. Speaks with animation and color. Expressive and dynamic.",
        "Sadaltager": "Male. Knowledgeable, learned, mid-pitched voice. Speaks thoughtfully and wisely. Intellectual tone.",
        "Zubenelgenubi": "Male. Casual, relaxed, low-pitched voice. Speaks informally and naturally. Easygoing and chill.",
        # 女性
        "Zephyr":     "Female. Bright, clear, high-pitched voice. Speaks with clarity and sparkle. Energetic and crisp.",
        "Kore":       "Female. Strong, firm, mid-pitched voice. Speaks with confidence and authority. Powerful and decisive.",
        "Leda":       "Female. Youthful, energetic, high-pitched voice. Speaks with vitality and enthusiasm. Fresh and lively.",
        "Aoede":      "Female. Breezy, natural, mid-pitched voice. Speaks in a relaxed, conversational way. Effortless and calm.",
        "Callirrhoe": "Female. Friendly, easy-going, low-to-mid pitched voice. Speaks warmly and casually. Approachable.",
        "Autonoe":    "Female. Bright, cheerful, high-pitched voice. Speaks with positivity and light. Uplifting tone.",
        "Despina":    "Female. Smooth, gentle, low-to-mid pitched voice. Speaks softly and elegantly. Polished and refined.",
        "Erinome":    "Female. Clear, articulate, mid-pitched voice. Speaks with precision. Professional and composed.",
        "Laomedeia":  "Female. Positive, upbeat, high-pitched voice. Speaks with optimism and cheer. Encouraging tone.",
        "Achernar":   "Female. Soft, warm, low-pitched voice. Speaks gently and soothingly. Intimate and comforting.",
        "Gacrux":     "Female. Mature, steady, mid-pitched voice. Speaks with gravitas and experience. Wise and grounded.",
        "Vindemiatrix":"Female. Gentle, delicate, low-pitched voice. Speaks softly and carefully. Tender and refined.",
        "Sulafat":    "Female. Warm, approachable, mid-pitched voice. Speaks with friendliness and care. Inviting tone.",
    }

    lines = ["This is a Japanese podcast translated from a foreign language.\n"]

    for speaker, voice in voice_mapping.items():
        direction = VOICE_DIRECTION.get(voice, f"Speaks naturally and clearly.")
        lines.append(f"{speaker}: {direction}\n")

    lines.append(
        "CRITICAL RULES:\n"
        "1. Each speaker MUST maintain their exact same voice from the very first word to the last. "
        "Do NOT change pitch, speed, or timbre at any point.\n"
        "2. Insert a natural pause (about 0.5 seconds of silence) between speaker turns.\n"
        "3. Do NOT rush. Speak at a relaxed, unhurried pace."
    )
    return "\n".join(lines)


# -------------------------------------------------------- チャンク分割 --

def _build_chunks(segments: list[dict], voice_mapping: dict[str, str],
                  custom_readings: dict[str, str]) -> list[dict]:
    """翻訳済みセグメントを TTS チャンクに分割する。

    Gemini TTS はマルチスピーカーが最大2人のため、
    1チャンク内の話者が2人を超えないよう分割する。
    """
    speech_direction = _build_speech_direction(voice_mapping)
    direction_len = len(speech_direction)

    chunks = []
    current_lines: list[str] = []
    current_chars = 0
    speakers_in_chunk: set[str] = set()
    prev_speaker = None

    def _flush():
        nonlocal current_lines, current_chars, speakers_in_chunk
        if current_lines:
            chunks.append({
                "text": speech_direction + "\n".join(current_lines),
                "has_multi_speaker": len(speakers_in_chunk) > 1,
                "speakers": speakers_in_chunk.copy(),
            })
            current_lines = []
            current_chars = 0
            speakers_in_chunk = set()

    for seg in segments:
        speaker = seg["speaker"]
        raw_text = seg.get("text", "").strip()
        if not raw_text:
            continue
        if speaker not in voice_mapping:
            continue

        text = _fix_readings(raw_text, custom_readings)
        formatted = f"{speaker}: {text}"

        total = direction_len + current_chars + len(formatted)

        # チャンクサイズ超過 or 3人目の話者 → フラッシュ
        needs_flush = False
        if total > CHUNK_MAX_CHARS and current_lines:
            needs_flush = True
        if speaker not in speakers_in_chunk and len(speakers_in_chunk) >= 2:
            needs_flush = True

        if needs_flush:
            _flush()
            prev_speaker = None

        if prev_speaker and prev_speaker != speaker:
            current_lines.append("[pause: 0.8s]")
            current_chars += len("[pause: 0.8s]")

        current_lines.append(formatted)
        current_chars += len(formatted)
        speakers_in_chunk.add(speaker)
        prev_speaker = speaker

    _flush()
    return chunks


# ----------------------------------------------------------- TTS 呼び出し --

def _tts_chunk(client: genai.Client, chunk: dict,
               voice_mapping: dict[str, str]) -> AudioSegment:
    text = chunk["text"]
    speakers = chunk.get("speakers", set())

    if chunk.get("has_multi_speaker"):
        speaker_configs = []
        for speaker_name in sorted(speakers):
            voice = voice_mapping.get(speaker_name, DEFAULT_VOICE)
            speaker_configs.append(
                types.SpeakerVoiceConfig(
                    speaker=speaker_name,
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice,
                        )
                    ),
                )
            )
        speech_config = types.SpeechConfig(
            multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                speaker_voice_configs=speaker_configs,
            )
        )
    else:
        speaker_name = next(iter(speakers)) if speakers else "Narrator"
        voice = voice_mapping.get(speaker_name, DEFAULT_VOICE)
        speech_config = types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=voice,
                )
            )
        )

    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=speech_config,
                ),
            )
            pcm_data = response.candidates[0].content.parts[0].inline_data.data
            return _pcm_to_audio_segment(pcm_data)
        except Exception as e:
            if attempt == 4:
                raise
            wait = 2 ** (attempt + 1)
            print(f"    リトライ {attempt+1}/5 ({wait}s待機): {str(e)[:120]}")
            time.sleep(wait)

    return AudioSegment.empty()


# ----------------------------------------------------------------- main --

def tts(segments: list[dict], output_name: str = "output",
        voice_features: dict[str, dict] | None = None) -> Path:
    """翻訳済みセグメントから MP3 を生成する。

    Args:
        segments: [{"speaker": "Speaker_1", "text": "..."}, ...]
        output_name: 出力ファイル名（拡張子なし）
        voice_features: 話者の声質分析結果（diarize.py から）

    Returns:
        生成した MP3 のパス
    """
    print(f"\n=== 音声生成開始 (Gemini Flash TTS) ===")

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY が .env に設定されていません")

    client = genai.Client(api_key=gemini_api_key)

    # ボイス自動割り当て（声質分析結果ベース）
    voice_mapping = _assign_voices(segments, voice_features)
    print(f"ボイス割り当て:")
    for speaker, voice in voice_mapping.items():
        print(f"  {speaker} → {voice}")

    # 読み辞書の自動更新
    print("\nカスタム読み辞書チェック中...")
    custom_readings = update_custom_readings(client, segments)

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # チャンク分割
    chunks = _build_chunks(segments, voice_mapping, custom_readings)
    print(f"\nチャンク数: {len(chunks)}")

    combined = AudioSegment.empty()

    for i, chunk in enumerate(chunks):
        speakers = ", ".join(sorted(chunk.get("speakers", set())))
        char_count = len(chunk["text"])
        text_preview = chunk["text"][:50].replace("\n", " ")
        print(f"  [{i+1}/{len(chunks)}] TTS ({speakers}, {char_count}文字): {text_preview}...")

        seg = _tts_chunk(client, chunk, voice_mapping)
        combined += seg
        time.sleep(API_CALL_INTERVAL)

    # 出力
    out_path = AUDIO_DIR / f"{output_name}.mp3"
    combined.export(out_path, format="mp3", bitrate="128k")

    duration_sec = len(combined) / 1000
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\n保存完了: {out_path}")
    print(f"  再生時間: {int(duration_sec // 60)}分{int(duration_sec % 60)}秒")
    print(f"  ファイルサイズ: {size_mb:.1f} MB")

    return out_path
