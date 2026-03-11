# transcribe.py - Whisper API で文字起こし

import io
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent.parent / ".env")

# Whisper API の 25MB 上限
WHISPER_MAX_BYTES = 24 * 1024 * 1024


def transcribe(mp3_bytes: bytes) -> str:
    """OpenAI Whisper API で音声を文字起こしする。

    言語は自動検出（language パラメータを指定しない）。
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY が .env に設定されていません")

    client = OpenAI(api_key=api_key)
    audio_file = io.BytesIO(mp3_bytes)
    audio_file.name = "episode.mp3"

    size_mb = len(mp3_bytes) / 1024 / 1024
    print(f"\nWhisper 文字起こし中... ({size_mb:.1f} MB)")

    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        # language を指定しない → 自動検出
    )

    text = transcript.text
    print(f"  文字起こし完了: {len(text)} 文字")
    return text
