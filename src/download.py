# download.py - URL から音声をダウンロード + メタデータ収集

import io
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import feedparser
import requests
from pydub import AudioSegment


WHISPER_MAX_BYTES = 24 * 1024 * 1024  # 24MB (Whisper API 上限は 25MB)


def _detect_url_type(url: str) -> str:
    """URL の種類を判定する。"""
    if re.search(r'(youtube\.com|youtu\.be)', url):
        return "youtube"
    if url.strip().endswith(".mp3") or "/audio/" in url:
        return "direct_audio"
    if any(kw in url for kw in ["feed", "rss", ".xml"]):
        return "rss"
    return "generic"


# -------------------------------------------------------- YouTube --

def _download_youtube(url: str, work_dir: Path) -> bytes:
    """yt-dlp で YouTube 動画の音声を MP3 としてダウンロードする。"""
    out_path = work_dir / "audio.mp3"
    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", "5",
        "-o", str(out_path),
        "--no-playlist",
        url,
    ]
    cookies_path = os.environ.get("YT_COOKIES_PATH")
    if cookies_path and Path(cookies_path).exists():
        cmd.insert(1, "--cookies")
        cmd.insert(2, cookies_path)
    print(f"  yt-dlp 実行中...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"  yt-dlp stderr: {result.stderr[:500]}")
        raise RuntimeError(f"yt-dlp failed: {result.stderr[:200]}")
    return out_path.read_bytes()


def _get_youtube_metadata(url: str) -> dict:
    """yt-dlp で YouTube のメタデータを取得する。"""
    cmd = [
        "yt-dlp",
        "--dump-json",
        "--no-playlist",
        "--skip-download",
        url,
    ]
    cookies_path = os.environ.get("YT_COOKIES_PATH")
    if cookies_path and Path(cookies_path).exists():
        cmd.insert(1, "--cookies")
        cmd.insert(2, cookies_path)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return {}
        data = json.loads(result.stdout)
        return {
            "title": data.get("title", ""),
            "description": data.get("description", ""),
            "channel": data.get("channel", "") or data.get("uploader", ""),
            "duration": data.get("duration"),
            "tags": data.get("tags", []),
        }
    except Exception as e:
        print(f"  メタデータ取得失敗: {e}")
        return {}


def _get_youtube_captions(url: str) -> str:
    """yt-dlp で YouTube の字幕を取得する。"""
    try:
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [
                "yt-dlp",
                "--write-auto-sub",
                "--write-sub",
                "--sub-lang", "en",
                "--sub-format", "vtt",
                "--skip-download",
                "-o", f"{tmp}/sub",
                "--no-playlist",
                url,
            ]
            cookies_path = os.environ.get("YT_COOKIES_PATH")
            if cookies_path and Path(cookies_path).exists():
                cmd.insert(1, "--cookies")
                cmd.insert(2, cookies_path)
            subprocess.run(cmd, capture_output=True, timeout=30)

            # 手動字幕を優先、なければ自動字幕
            for suffix in [".en.vtt", ".en-orig.vtt"]:
                vtt_path = Path(tmp) / f"sub{suffix}"
                if vtt_path.exists():
                    return _parse_vtt(vtt_path.read_text())

            # どんな .vtt でも拾う
            for vtt_file in Path(tmp).glob("*.vtt"):
                return _parse_vtt(vtt_file.read_text())

    except Exception as e:
        print(f"  字幕取得失敗: {e}")

    return ""


def _parse_vtt(text: str) -> str:
    """VTT 字幕をプレーンテキストに変換する。"""
    lines = []
    prev_line = ""
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("WEBVTT") or line.startswith("NOTE"):
            continue
        if re.match(r'^\d+$', line):
            continue
        if re.match(r'\d{2}:\d{2}', line):
            continue
        # HTML タグ・タイムスタンプタグを除去
        cleaned = re.sub(r'<[^>]+>', '', line)
        cleaned = cleaned.strip()
        if cleaned and cleaned != prev_line:
            lines.append(cleaned)
            prev_line = cleaned
    return " ".join(lines)


# -------------------------------------------------------- Podcast --

def _download_direct_audio(url: str) -> bytes:
    """直接音声URLからダウンロードする。"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
    }
    print(f"  音声ダウンロード中...")
    resp = requests.get(url, timeout=300, stream=True, headers=headers,
                        allow_redirects=True)
    resp.raise_for_status()

    chunks = []
    received = 0
    for chunk in resp.iter_content(chunk_size=1024 * 1024):
        chunks.append(chunk)
        received += len(chunk)
        if received >= WHISPER_MAX_BYTES * 2:
            print(f"  ダウンロード上限到達 ({received // 1024 // 1024} MB)")
            break

    return b"".join(chunks)


def _download_from_rss(url: str) -> tuple[bytes, dict]:
    """RSS フィードから最新エピソードの音声をダウンロードする。"""
    print(f"  RSS フィード解析中...")
    feed = feedparser.parse(url)
    if not feed.entries:
        raise ValueError("RSS フィードにエピソードが見つかりません")

    entry = feed.entries[0]
    mp3_url = None
    for enc in entry.get("enclosures", []):
        if "audio" in enc.get("type", "") or enc.get("url", "").endswith(".mp3"):
            mp3_url = enc.get("url")
            break

    if not mp3_url:
        raise ValueError(f"RSS エピソードに音声URLが見つかりません: {entry.get('title', '?')}")

    title = entry.get("title", "Unknown Episode")
    print(f"  エピソード: {title}")

    audio_bytes = _download_direct_audio(mp3_url)

    # RSS からメタデータを充実させる
    # 番組レベルの情報
    show_title = feed.feed.get("title", "")
    show_author = feed.feed.get("author", "") or feed.feed.get("itunes_author", "")
    show_description = feed.feed.get("subtitle", "") or feed.feed.get("summary", "")

    # エピソードレベルの情報
    episode_description = entry.get("summary", "") or entry.get("content", [{}])[0].get("value", "")

    metadata = {
        "title": title,
        "description": episode_description[:2000],
        "url": entry.get("link", url),
        "published_at": entry.get("published", ""),
        "show_title": show_title,
        "show_author": show_author,
        "show_description": show_description[:500],
    }
    return audio_bytes, metadata


# -------------------------------------------------------- 共通 --

def _trim_mp3_bytes(mp3_bytes: bytes, max_seconds: int) -> bytes:
    """MP3 を指定秒数にトリミングする（Whisper 上限対策）。"""
    audio = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
    trimmed = audio[:max_seconds * 1000]
    buf = io.BytesIO()
    trimmed.export(buf, format="mp3", bitrate="64k")
    return buf.getvalue()


def _ensure_size(mp3_bytes: bytes) -> bytes:
    """Whisper API の上限を超える場合はトリミングする。"""
    if len(mp3_bytes) <= WHISPER_MAX_BYTES:
        return mp3_bytes
    total_mb = len(mp3_bytes) / 1024 / 1024
    max_sec = int(WHISPER_MAX_BYTES * 8 / 64000)
    print(f"  音声が {total_mb:.1f} MB。Whisper上限のため {max_sec // 60} 分にトリミング...")
    return _trim_mp3_bytes(mp3_bytes, max_sec)


# -------------------------------------------------------- メイン --

def download(url: str) -> tuple[bytes, dict]:
    """URL から音声をダウンロードし、(mp3_bytes, metadata) を返す。

    metadata には title, description, captions 等を含む。
    """
    url_type = _detect_url_type(url)
    print(f"\nURL種別: {url_type}")
    print(f"URL: {url}")

    if url_type == "rss":
        mp3_bytes, metadata = _download_from_rss(url)
        metadata["source_type"] = "podcast"
        return _ensure_size(mp3_bytes), metadata

    if url_type == "youtube" or url_type == "generic":
        # メタデータと字幕を取得
        yt_meta = _get_youtube_metadata(url)
        captions = _get_youtube_captions(url) if url_type == "youtube" else ""

        with tempfile.TemporaryDirectory() as tmp:
            mp3_bytes = _download_youtube(url, Path(tmp))

        metadata = {
            "title": yt_meta.get("title", "Unknown"),
            "description": yt_meta.get("description", "")[:2000],
            "channel": yt_meta.get("channel", ""),
            "duration": yt_meta.get("duration"),
            "tags": yt_meta.get("tags", []),
            "captions": captions[:3000] if captions else "",
            "url": url,
            "source_type": "youtube" if url_type == "youtube" else "video",
        }

        if captions:
            print(f"  字幕取得: {len(captions)} 文字")
        if yt_meta.get("description"):
            print(f"  説明文取得: {len(yt_meta['description'])} 文字")

        return _ensure_size(mp3_bytes), metadata

    # direct_audio
    mp3_bytes = _download_direct_audio(url)
    metadata = {
        "title": "Audio",
        "url": url,
        "source_type": "direct_audio",
    }
    return _ensure_size(mp3_bytes), metadata
