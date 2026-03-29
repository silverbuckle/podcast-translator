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
    if "podcasts.apple.com" in url:
        return "apple_podcast"
    if url.strip().endswith(".mp3") or "/audio/" in url:
        return "direct_audio"
    if any(kw in url for kw in ["feed", "rss", ".xml"]):
        return "rss"
    return "generic"


# -------------------------------------------------------- YouTube --

def _yt_dlp_extra_args() -> list[str]:
    """yt-dlp の共通オプション（cookies, JS ランタイム）を返す。"""
    return ["--cookies-from-browser", "chrome"]


def _subprocess_env() -> dict:
    """subprocess 用の環境変数。node 等の JS ランタイムを検出できるよう PATH を補完する。"""
    env = os.environ.copy()
    for p in ("/opt/homebrew/bin", "/usr/local/bin"):
        if p not in env.get("PATH", ""):
            env["PATH"] = p + ":" + env.get("PATH", "")
    return env


def _download_youtube(url: str, work_dir: Path) -> bytes:
    """yt-dlp で YouTube 動画の音声を MP3 としてダウンロードする。"""
    out_path = work_dir / "audio.mp3"
    cmd = [
        "yt-dlp",
        *_yt_dlp_extra_args(),
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", "5",
        "-o", str(out_path),
        "--no-playlist",
        url,
    ]
    print(f"  yt-dlp 実行中...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                            env=_subprocess_env())
    if result.returncode != 0:
        print(f"  yt-dlp stderr: {result.stderr[:500]}")
        raise RuntimeError(f"yt-dlp failed: {result.stderr[:200]}")
    return out_path.read_bytes()


def _get_youtube_metadata(url: str) -> dict:
    """yt-dlp で YouTube のメタデータを取得する。"""
    cmd = [
        "yt-dlp",
        *_yt_dlp_extra_args(),
        "--dump-json",
        "--no-playlist",
        "--skip-download",
        url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                                env=_subprocess_env())
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
                *_yt_dlp_extra_args(),
                "--write-auto-sub",
                "--write-sub",
                "--sub-lang", "en",
                "--sub-format", "vtt",
                "--skip-download",
                "-o", f"{tmp}/sub",
                "--no-playlist",
                url,
            ]
            subprocess.run(cmd, capture_output=True, timeout=30,
                            env=_subprocess_env())

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
        if received >= 500 * 1024 * 1024:  # 500MB 安全上限
            print(f"  ダウンロード上限到達 ({received // 1024 // 1024} MB)")
            break

    return b"".join(chunks)


def _download_from_rss(url: str) -> tuple[bytes, dict]:
    """RSS フィードから最新エピソードの音声をダウンロードする。"""
    audio_url, metadata = _find_episode_in_rss(url)
    print(f"  エピソード: {metadata.get('title', '?')}")
    audio_bytes = _download_direct_audio(audio_url)
    return audio_bytes, metadata


# -------------------------------------------------------- Apple Podcast --

def _parse_apple_podcast_url(url: str) -> tuple[str | None, str | None]:
    """Apple Podcast URL から podcast_id と episode_id を抽出する。"""
    # https://podcasts.apple.com/us/podcast/episode-name/id1073226719?i=1000753545687
    podcast_id_match = re.search(r'/id(\d+)', url)
    episode_id_match = re.search(r'[?&]i=(\d+)', url)
    podcast_id = podcast_id_match.group(1) if podcast_id_match else None
    episode_id = episode_id_match.group(1) if episode_id_match else None
    return podcast_id, episode_id


def _lookup_apple_episode(podcast_id: str, episode_id: str | None) -> dict:
    """iTunes Lookup API でエピソード情報を取得する。

    Returns:
        {"title", "description", "audio_url", "episode_guid", "feed_url", ...}
    """
    # まずポッドキャスト情報を取得
    api_url = f"https://itunes.apple.com/lookup?id={podcast_id}&media=podcast"
    resp = requests.get(api_url, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [])
    if not results:
        raise ValueError(f"iTunes API でポッドキャストが見つかりません (id={podcast_id})")

    podcast_info = results[0]
    feed_url = podcast_info.get("feedUrl", "")
    show_title = podcast_info.get("collectionName", "")
    show_author = podcast_info.get("artistName", "")

    if not episode_id:
        return {"feed_url": feed_url, "show_title": show_title,
                "show_author": show_author}

    # エピソード一覧を取得（最大200件）
    print(f"  iTunes API でエピソード検索中...")
    ep_url = (f"https://itunes.apple.com/lookup?id={podcast_id}"
              f"&media=podcast&entity=podcastEpisode&limit=200")
    resp = requests.get(ep_url, timeout=30)
    resp.raise_for_status()
    ep_data = resp.json()

    # trackId でマッチング
    for item in ep_data.get("results", []):
        if str(item.get("trackId", "")) == episode_id:
            print(f"  iTunes API でエピソード発見: {item.get('trackName', '?')}")
            return {
                "title": item.get("trackName", ""),
                "description": item.get("description", "")[:2000],
                "audio_url": item.get("episodeUrl", ""),
                "episode_guid": item.get("episodeGuid", ""),
                "feed_url": feed_url,
                "show_title": show_title,
                "show_author": show_author,
                "published_at": item.get("releaseDate", ""),
            }

    print(f"  iTunes API にエピソード {episode_id} が見つかりません（古いエピソードの可能性）")
    return {"feed_url": feed_url, "show_title": show_title,
            "show_author": show_author}


def _find_episode_in_rss(feed_url: str, episode_guid: str | None = None) -> tuple[str, dict]:
    """RSS フィードからエピソードを探し、(audio_url, entry_metadata) を返す。"""
    print(f"  RSS フィード解析中...")
    feed = feedparser.parse(feed_url)
    if not feed.entries:
        raise ValueError("RSS フィードにエピソードが見つかりません")

    target_entry = None
    if episode_guid:
        for entry in feed.entries:
            guid = entry.get("id", "") or entry.get("guid", "")
            if guid == episode_guid:
                target_entry = entry
                print(f"  RSS で guid マッチ: {entry.get('title', '?')}")
                break

    if not target_entry:
        if episode_guid:
            print(f"  RSS で guid が見つかりません。最新エピソードを使用します。")
        target_entry = feed.entries[0]

    # 音声 URL を取得
    mp3_url = None
    for enc in target_entry.get("enclosures", []):
        if "audio" in enc.get("type", "") or enc.get("url", "").endswith(".mp3"):
            mp3_url = enc.get("url")
            break
    if not mp3_url:
        for link in target_entry.get("links", []):
            if "audio" in link.get("type", ""):
                mp3_url = link.get("href")
                break
    if not mp3_url:
        raise ValueError(f"エピソードに音声 URL が見つかりません: {target_entry.get('title', '?')}")

    show_title = feed.feed.get("title", "")
    show_author = feed.feed.get("author", "") or feed.feed.get("itunes_author", "")
    show_description = feed.feed.get("subtitle", "") or feed.feed.get("summary", "")
    episode_description = (target_entry.get("summary", "")
                           or target_entry.get("content", [{}])[0].get("value", ""))

    title = target_entry.get("title", "Unknown Episode")
    entry_meta = {
        "title": title,
        "description": episode_description[:2000],
        "url": target_entry.get("link", feed_url),
        "published_at": target_entry.get("published", ""),
        "show_title": show_title,
        "show_author": show_author,
        "show_description": show_description[:500],
    }
    return mp3_url, entry_meta


def _download_apple_podcast(url: str) -> tuple[bytes, dict]:
    """Apple Podcast URL から音声をダウンロードする。

    1. iTunes Lookup API でエピソード情報を取得（audio_url, episode_guid）
    2. audio_url があればそのまま使用、なければ RSS から検索
    """
    podcast_id, episode_id = _parse_apple_podcast_url(url)
    if not podcast_id:
        raise ValueError(f"Apple Podcast URL からポッドキャスト ID を抽出できません: {url}")

    print(f"  Podcast ID: {podcast_id}, Episode ID: {episode_id or '(最新)'}")

    # iTunes Lookup API でエピソード情報を取得
    ep_info = _lookup_apple_episode(podcast_id, episode_id)
    feed_url = ep_info.get("feed_url", "")

    audio_url = ep_info.get("audio_url", "")
    metadata = {}

    if audio_url:
        # iTunes API から直接音声 URL を取得できた
        print(f"  エピソード: {ep_info.get('title', '?')}")
        metadata = {
            "title": ep_info.get("title", "Unknown"),
            "description": ep_info.get("description", "")[:2000],
            "url": url,
            "published_at": ep_info.get("published_at", ""),
            "show_title": ep_info.get("show_title", ""),
            "show_author": ep_info.get("show_author", ""),
        }
    elif feed_url:
        # RSS フィードから検索
        print(f"  RSS フィード: {feed_url}")
        episode_guid = ep_info.get("episode_guid")
        audio_url, metadata = _find_episode_in_rss(feed_url, episode_guid)
    else:
        raise ValueError("音声 URL も RSS フィード URL も取得できません")

    audio_bytes = _download_direct_audio(audio_url)
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
    """音声サイズを確認する。チャンク分割で Whisper 上限に対応するためトリミングは行わない。"""
    total_mb = len(mp3_bytes) / 1024 / 1024
    if total_mb > WHISPER_MAX_BYTES / 1024 / 1024:
        print(f"  音声サイズ: {total_mb:.1f} MB（チャンク分割で処理）")
    return mp3_bytes


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

    if url_type == "apple_podcast":
        mp3_bytes, metadata = _download_apple_podcast(url)
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
