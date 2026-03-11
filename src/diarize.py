# diarize.py - 話者分離 + Whisper タイムスタンプ照合
# 長い音声は20分チャンクに分割して処理

import io
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment
from pyannote.audio import Pipeline

load_dotenv(Path(__file__).parent.parent / ".env")

# チャンク分割の設定
CHUNK_DURATION_MS = 20 * 60 * 1000   # 20分
OVERLAP_MS = 30 * 1000                # 30秒オーバーラップ
CHUNK_THRESHOLD_MS = 22 * 60 * 1000   # 22分以上でチャンク分割


def _load_pipeline():
    """pyannote パイプラインをロードする（1回だけ）。"""
    hf_token = os.getenv("HF_AUTH_TOKEN")
    if not hf_token:
        raise ValueError("HF_AUTH_TOKEN が .env に設定されていません（pyannote-audio 用）")

    print("  pyannote-audio パイプラインロード中...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )

    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        pipeline.to(torch.device("mps"))

    return pipeline


def _diarize(audio_path: str, min_speakers: int | None = None,
             max_speakers: int | None = None,
             pipeline=None) -> list[dict]:
    """pyannote-audio で話者分離する。"""
    if pipeline is None:
        pipeline = _load_pipeline()

    print("  pyannote-audio 話者分離中...")

    kwargs = {}
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    result = pipeline(audio_path, **kwargs)
    diarization = getattr(result, "speaker_diarization", result)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": round(turn.start, 2),
            "end": round(turn.end, 2),
        })

    # 話者名を簡潔にリネーム（SPEAKER_00 → Speaker_1）
    speaker_map = {}
    for seg in segments:
        if seg["speaker"] not in speaker_map:
            speaker_map[seg["speaker"]] = f"Speaker_{len(speaker_map) + 1}"
        seg["speaker"] = speaker_map[seg["speaker"]]

    print(f"  話者数: {len(speaker_map)}")
    print(f"  セグメント数: {len(segments)}")
    for name, label in speaker_map.items():
        total = sum(s["end"] - s["start"] for s in segments if s["speaker"] == label)
        print(f"    {label} ({name}): {total:.1f}秒")

    return segments


def _whisper_with_timestamps(mp3_bytes: bytes) -> list[dict]:
    """Whisper API でタイムスタンプ付き文字起こしする。"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY が .env に設定されていません")

    client = OpenAI(api_key=api_key)
    audio_file = io.BytesIO(mp3_bytes)
    audio_file.name = "episode.mp3"

    size_mb = len(mp3_bytes) / 1024 / 1024
    print(f"\n  Whisper 文字起こし中 (タイムスタンプ付き, {size_mb:.1f} MB)...")

    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="verbose_json",
        timestamp_granularities=["segment"],
    )

    segments = []
    for seg in transcript.segments:
        segments.append({
            "text": seg.text.strip() if hasattr(seg, "text") else seg["text"].strip(),
            "start": seg.start if hasattr(seg, "start") else seg["start"],
            "end": seg.end if hasattr(seg, "end") else seg["end"],
        })

    print(f"  Whisper セグメント数: {len(segments)}")
    return segments


def _align_speakers(whisper_segments: list[dict],
                    diarization_segments: list[dict]) -> list[dict]:
    """Whisper のテキストセグメントに話者ラベルを付与する。"""
    result = []
    for ws in whisper_segments:
        ws_start = ws["start"]
        ws_end = ws["end"]
        ws_duration = ws_end - ws_start

        if ws_duration <= 0:
            continue

        speaker_overlap: dict[str, float] = {}
        for ds in diarization_segments:
            overlap_start = max(ws_start, ds["start"])
            overlap_end = min(ws_end, ds["end"])
            overlap = max(0, overlap_end - overlap_start)
            if overlap > 0:
                speaker = ds["speaker"]
                speaker_overlap[speaker] = speaker_overlap.get(speaker, 0) + overlap

        if speaker_overlap:
            speaker = max(speaker_overlap, key=speaker_overlap.get)
        else:
            speaker = _nearest_speaker(ws_start, diarization_segments)

        result.append({
            "speaker": speaker,
            "text": ws["text"],
            "start": ws_start,
            "end": ws_end,
        })

    return result


def _nearest_speaker(timestamp: float, diarization_segments: list[dict]) -> str:
    """指定タイムスタンプに最も近い話者を返す。"""
    if not diarization_segments:
        return "Speaker_1"

    best = diarization_segments[0]
    best_dist = abs(timestamp - best["start"])
    for ds in diarization_segments:
        dist = min(abs(timestamp - ds["start"]), abs(timestamp - ds["end"]))
        if dist < best_dist:
            best = ds
            best_dist = dist
    return best["speaker"]


def _merge_consecutive(segments: list[dict]) -> list[dict]:
    """同じ話者の連続セグメントを結合する。"""
    if not segments:
        return []

    merged = [segments[0].copy()]
    for seg in segments[1:]:
        if seg["speaker"] == merged[-1]["speaker"]:
            merged[-1]["text"] += " " + seg["text"]
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(seg.copy())

    return merged


def _estimate_f0_autocorr(samples: np.ndarray, sample_rate: int) -> float:
    """自己相関法で基本周波数 (F0) を推定する。"""
    window_size = int(0.03 * sample_rate)
    hop_size = int(0.01 * sample_rate)
    min_lag = int(sample_rate / 400)
    max_lag = int(sample_rate / 70)

    f0_values = []

    for start in range(0, len(samples) - window_size, hop_size):
        frame = samples[start:start + window_size]

        if np.max(np.abs(frame)) < np.max(np.abs(samples)) * 0.05:
            continue

        frame = frame - np.mean(frame)
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr) // 2:]

        if max_lag >= len(corr):
            continue

        search_region = corr[min_lag:max_lag]
        if len(search_region) == 0:
            continue

        peak_idx = np.argmax(search_region) + min_lag

        if corr[peak_idx] > 0.3 * corr[0]:
            f0 = sample_rate / peak_idx
            f0_values.append(f0)

    if not f0_values:
        return 0.0

    f0_arr = np.array(f0_values)
    median = np.median(f0_arr)
    filtered = f0_arr[np.abs(f0_arr - median) < median * 0.3]
    return float(np.median(filtered)) if len(filtered) > 0 else float(median)


def _analyze_voice(mp3_bytes: bytes, segments: list[dict],
                   diarization_segments: list[dict]) -> dict[str, dict]:
    """各話者の声質特徴を分析する（自己相関法による F0 推定）。"""
    try:
        audio = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        sample_rate = audio.frame_rate

        speakers = sorted(set(s["speaker"] for s in segments))
        voice_features = {}

        for speaker in speakers:
            speaker_ds = [ds for ds in diarization_segments if ds["speaker"] == speaker]
            if not speaker_ds:
                continue

            speaker_samples = []
            for ds in speaker_ds:
                start_idx = int(ds["start"] * sample_rate)
                end_idx = int(ds["end"] * sample_rate)
                if end_idx <= len(samples):
                    speaker_samples.append(samples[start_idx:end_idx])

            if not speaker_samples:
                continue

            combined = np.concatenate(speaker_samples)
            rms = np.sqrt(np.mean(combined ** 2))

            f0 = _estimate_f0_autocorr(combined, sample_rate)

            if f0 < 150:
                pitch = "low"
                gender_hint = "male"
            elif f0 < 200:
                pitch = "mid"
                gender_hint = "male"
            elif f0 < 260:
                pitch = "mid-high"
                gender_hint = "female"
            else:
                pitch = "high"
                gender_hint = "female"

            voice_features[speaker] = {
                "pitch": pitch,
                "energy": "high" if rms > 2000 else "moderate" if rms > 500 else "low",
                "gender_hint": gender_hint,
                "estimated_f0_hz": round(f0),
            }

            print(f"    {speaker}: F0={f0:.0f}Hz → pitch={pitch}, "
                  f"gender_hint={gender_hint}, "
                  f"energy={voice_features[speaker]['energy']}")

        return voice_features

    except Exception as e:
        print(f"  警告: 声質分析に失敗: {e}")
        return {}


def _merge_similar_speakers(segments: list[dict], diarization_segments: list[dict],
                            mp3_bytes: bytes) -> tuple[list[dict], list[dict]]:
    """F0 が近い話者を統合して過剰分割を修正する。"""
    audio = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    sample_rate = audio.frame_rate

    speakers = sorted(set(s["speaker"] for s in diarization_segments))
    if len(speakers) <= 2:
        return segments, diarization_segments

    speaker_f0: dict[str, float] = {}
    for speaker in speakers:
        speaker_ds = [ds for ds in diarization_segments if ds["speaker"] == speaker]
        speaker_samples = []
        for ds in speaker_ds:
            start_idx = int(ds["start"] * sample_rate)
            end_idx = int(ds["end"] * sample_rate)
            if end_idx <= len(samples):
                speaker_samples.append(samples[start_idx:end_idx])
        if speaker_samples:
            combined = np.concatenate(speaker_samples)
            speaker_f0[speaker] = _estimate_f0_autocorr(combined, sample_rate)

    merge_map: dict[str, str] = {}
    sorted_speakers = sorted(speaker_f0.keys(), key=lambda s: speaker_f0[s])

    for i, sp in enumerate(sorted_speakers):
        if sp in merge_map:
            continue
        merge_map[sp] = sp
        for j in range(i + 1, len(sorted_speakers)):
            other = sorted_speakers[j]
            if other in merge_map:
                continue
            if abs(speaker_f0[sp] - speaker_f0[other]) < 30:
                merge_map[other] = sp
                print(f"    統合: {other} (F0={speaker_f0[other]:.0f}Hz) → "
                      f"{sp} (F0={speaker_f0[sp]:.0f}Hz)")

    if len(set(merge_map.values())) == len(speakers):
        return segments, diarization_segments

    new_segments = []
    for seg in segments:
        new_seg = seg.copy()
        new_seg["speaker"] = merge_map.get(seg["speaker"], seg["speaker"])
        new_segments.append(new_seg)

    new_diar = []
    for ds in diarization_segments:
        new_ds = ds.copy()
        new_ds["speaker"] = merge_map.get(ds["speaker"], ds["speaker"])
        new_diar.append(new_ds)

    unique_speakers = sorted(set(merge_map.values()))
    rename_map = {sp: f"Speaker_{i+1}" for i, sp in enumerate(unique_speakers)}
    for seg in new_segments:
        seg["speaker"] = rename_map[seg["speaker"]]
    for ds in new_diar:
        ds["speaker"] = rename_map[ds["speaker"]]

    merged_count = len(speakers) - len(unique_speakers)
    print(f"    話者統合: {len(speakers)}人 → {len(unique_speakers)}人 "
          f"({merged_count}人を統合)")

    return _merge_consecutive(new_segments), new_diar


# -------------------------------------------------------- チャンク分割処理 --

def _split_audio_chunks(audio: AudioSegment) -> list[tuple[AudioSegment, int]]:
    """音声を20分チャンクに分割する。

    Returns:
        [(chunk_audio, offset_ms), ...]
        offset_ms: このチャンクの有効領域の開始位置（元音声での絶対位置）
    """
    total_ms = len(audio)
    if total_ms <= CHUNK_DURATION_MS:
        return [(audio, 0)]

    chunks = []
    pos = 0
    while pos < total_ms:
        # チャンクの切り出し範囲（前後にオーバーラップを含む）
        chunk_start = max(0, pos - OVERLAP_MS) if pos > 0 else 0
        chunk_end = min(total_ms, pos + CHUNK_DURATION_MS + OVERLAP_MS)

        chunk = audio[chunk_start:chunk_end]
        chunks.append((chunk, pos))

        pos += CHUNK_DURATION_MS

    return chunks


def _chunk_to_mp3_bytes(chunk: AudioSegment) -> bytes:
    """AudioSegment を MP3 バイトに変換する（Whisper 用）。"""
    buf = io.BytesIO()
    chunk.export(buf, format="mp3", bitrate="64k")
    return buf.getvalue()


def _compute_speaker_f0(audio: AudioSegment,
                        diar_segments: list[dict]) -> dict[str, float]:
    """話者ごとの F0 を計算する。"""
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    sample_rate = audio.frame_rate

    speakers = sorted(set(s["speaker"] for s in diar_segments))
    f0_map = {}

    for speaker in speakers:
        speaker_ds = [ds for ds in diar_segments if ds["speaker"] == speaker]
        speaker_samples = []
        for ds in speaker_ds:
            start_idx = int(ds["start"] * sample_rate)
            end_idx = int(ds["end"] * sample_rate)
            if end_idx <= len(samples):
                speaker_samples.append(samples[start_idx:end_idx])
        if speaker_samples:
            combined = np.concatenate(speaker_samples)
            if len(combined) > sample_rate * 2:  # 最低2秒分必要
                f0_map[speaker] = _estimate_f0_autocorr(combined, sample_rate)

    return f0_map


def _unify_speakers_across_chunks(
    chunk_results: list[tuple[list[dict], dict[str, float]]]
) -> tuple[list[dict], list[dict]]:
    """チャンク間で話者ラベルを統一する。

    最初のチャンクの話者を基準とし、後続チャンクの話者を
    F0 の近さで基準話者にマッチングする。

    Returns:
        (unified_segments, unified_diar_segments)
    """
    if len(chunk_results) <= 1:
        segs, diar, _ = chunk_results[0] if chunk_results else ([], [], {})
        return segs, diar

    # 最初のチャンクの話者 F0 を基準にする
    _, _, canonical_f0 = chunk_results[0]
    canonical_speakers = {f0: speaker for speaker, f0
                          in canonical_f0.items() if f0 > 0}

    all_segments = []
    all_diar = []

    for chunk_idx, (segs, diar, f0_map) in enumerate(chunk_results):
        if chunk_idx == 0:
            all_segments.extend(segs)
            all_diar.extend(diar)
            continue

        # このチャンクの話者を基準話者にマッチング
        rename_map: dict[str, str] = {}
        used_canonical: set[str] = set()

        for local_speaker, local_f0 in sorted(f0_map.items(),
                                               key=lambda x: x[1]):
            if local_f0 <= 0:
                continue
            best_match = None
            best_diff = float('inf')
            for can_f0, can_speaker in canonical_speakers.items():
                if can_speaker in used_canonical:
                    continue
                diff = abs(local_f0 - can_f0)
                if diff < best_diff and diff < 50:  # 50Hz 以内でマッチ
                    best_diff = diff
                    best_match = can_speaker

            if best_match:
                rename_map[local_speaker] = best_match
                used_canonical.add(best_match)
            else:
                # 新しい話者として追加
                new_name = f"Speaker_{len(canonical_speakers) + 1}"
                rename_map[local_speaker] = new_name
                if local_f0 > 0:
                    canonical_speakers[local_f0] = new_name

        # ラベルを置換
        for seg in segs:
            new_seg = seg.copy()
            new_seg["speaker"] = rename_map.get(seg["speaker"], seg["speaker"])
            all_segments.append(new_seg)
        for ds in diar:
            new_ds = ds.copy()
            new_ds["speaker"] = rename_map.get(ds["speaker"], ds["speaker"])
            all_diar.append(new_ds)

        if rename_map:
            print(f"    チャンク {chunk_idx + 1} 話者マッピング: "
                  + ", ".join(f"{k}→{v}" for k, v in rename_map.items()))

    return all_segments, all_diar


def _deduplicate_overlap(segments: list[dict],
                         chunk_boundaries: list[int]) -> list[dict]:
    """オーバーラップ領域の重複セグメントを除去する。

    各チャンク境界の前後 OVERLAP_MS/1000 秒は重複があるため、
    後のチャンクのセグメントを削除する。
    """
    if not chunk_boundaries:
        return segments

    overlap_sec = OVERLAP_MS / 1000
    result = []

    for seg in segments:
        skip = False
        for boundary_ms in chunk_boundaries:
            boundary_sec = boundary_ms / 1000
            # このセグメントが境界のオーバーラップ領域にあり、
            # かつ境界より前（= 前のチャンクからのもの）でない場合はスキップ
            if (boundary_sec - overlap_sec < seg["start"] < boundary_sec
                    and seg["end"] > boundary_sec):
                # 境界をまたぐセグメント → 保持
                pass
            elif (boundary_sec <= seg["start"] < boundary_sec + overlap_sec
                  and any(r["start"] <= seg["start"] and r["end"] >= seg["end"]
                          and r is not seg for r in result)):
                skip = True
                break
        if not skip:
            result.append(seg)

    return result


def _process_chunked(mp3_bytes: bytes, audio: AudioSegment,
                     min_speakers: int | None = None,
                     max_speakers: int | None = None,
                     ) -> tuple[list[dict], dict[str, dict]]:
    """長い音声をチャンク分割して処理する。"""
    duration_min = len(audio) / 1000 / 60
    print(f"\n  長い音声 ({duration_min:.0f}分) → チャンク分割処理")

    # パイプラインを1回だけロード
    pipeline = _load_pipeline()

    # チャンク分割
    chunks = _split_audio_chunks(audio)
    print(f"  チャンク数: {len(chunks)}")

    chunk_results = []  # [(segments, diar_segments, f0_map), ...]
    chunk_boundaries = []  # チャンク境界位置 (ms)

    for i, (chunk_audio, offset_ms) in enumerate(chunks):
        offset_sec = offset_ms / 1000
        chunk_dur = len(chunk_audio) / 1000
        print(f"\n  --- チャンク {i + 1}/{len(chunks)} "
              f"({offset_sec:.0f}s ~ {offset_sec + chunk_dur:.0f}s) ---")

        if i > 0:
            chunk_boundaries.append(offset_ms)

        # WAV に変換して pyannote に渡す
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            chunk_audio.export(f, format="wav")
            chunk_wav_path = f.name

        try:
            # 話者分離
            diar_segments = _diarize(
                chunk_wav_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                pipeline=pipeline,
            )

            # Whisper 文字起こし
            chunk_mp3 = _chunk_to_mp3_bytes(chunk_audio)
            whisper_segments = _whisper_with_timestamps(chunk_mp3)

            # 話者ラベル照合
            aligned = _align_speakers(whisper_segments, diar_segments)

            # タイムスタンプをオフセット分だけ補正
            # ただし前にオーバーラップがある場合はその分を考慮
            actual_start_ms = max(0, offset_ms - OVERLAP_MS) if offset_ms > 0 else 0
            time_offset = actual_start_ms / 1000

            for seg in aligned:
                seg["start"] = round(seg["start"] + time_offset, 2)
                seg["end"] = round(seg["end"] + time_offset, 2)
            for ds in diar_segments:
                ds["start"] = round(ds["start"] + time_offset, 2)
                ds["end"] = round(ds["end"] + time_offset, 2)

            # チャンク内の話者 F0 を計算
            f0_map = _compute_speaker_f0(chunk_audio, [
                {"speaker": ds["speaker"],
                 "start": ds["start"] - time_offset,
                 "end": ds["end"] - time_offset}
                for ds in diar_segments
            ])

            chunk_results.append((aligned, diar_segments, f0_map))

        finally:
            Path(chunk_wav_path).unlink(missing_ok=True)

    # チャンク間の話者ラベルを統一
    print("\n  チャンク間話者統一中...")
    all_segments, all_diar = _unify_speakers_across_chunks(chunk_results)

    # オーバーラップ領域の重複除去
    all_segments = _deduplicate_overlap(all_segments, chunk_boundaries)

    # 時間順にソート
    all_segments.sort(key=lambda s: s["start"])
    all_diar.sort(key=lambda s: s["start"])

    # 連続セグメント結合
    merged = _merge_consecutive(all_segments)
    print(f"\n  全チャンク結合後セグメント数: {len(merged)}")

    # F0 ベースで過剰分割された話者を統合
    print("\n  話者統合チェック中...")
    merged, all_diar = _merge_similar_speakers(merged, all_diar, mp3_bytes)

    # 声質分析
    print("\n  声質分析中...")
    voice_features = _analyze_voice(mp3_bytes, merged, all_diar)

    return merged, voice_features


# -------------------------------------------------------- メインエントリポイント --

def diarize_and_transcribe(mp3_bytes: bytes,
                          min_speakers: int | None = None,
                          max_speakers: int | None = None,
                          ) -> tuple[list[dict], dict[str, dict]]:
    """話者分離 + 文字起こし + 照合を行う。

    22分以上の音声は自動的に20分チャンクに分割して処理する。

    Args:
        mp3_bytes: MP3 音声データ
        min_speakers: 最小話者数（メタデータ分析結果から）
        max_speakers: 最大話者数

    Returns:
        (segments, voice_features)
    """
    if min_speakers:
        print(f"  話者数ヒント: {min_speakers}人")

    # WAV 16kHz mono に変換
    audio = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
    audio = audio.set_frame_rate(16000).set_channels(1)

    # 長い音声はチャンク分割
    if len(audio) > CHUNK_THRESHOLD_MS:
        return _process_chunked(mp3_bytes, audio, min_speakers, max_speakers)

    # 短い音声は従来通り
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        audio.export(f, format="wav")
        tmp_path = f.name

    try:
        diarization_segments = _diarize(
            tmp_path, min_speakers=min_speakers, max_speakers=max_speakers
        )

        if len(set(s["speaker"] for s in diarization_segments)) <= 1:
            print("  話者が1人のみ → 話者分離スキップ")
            whisper_segments = _whisper_with_timestamps(mp3_bytes)
            segments = [{"speaker": "Narrator", "text": s["text"],
                         "start": s["start"], "end": s["end"]}
                        for s in whisper_segments]
            return _merge_consecutive(segments), {}

        whisper_segments = _whisper_with_timestamps(mp3_bytes)

        print("\n  話者ラベル照合中...")
        aligned = _align_speakers(whisper_segments, diarization_segments)

        merged = _merge_consecutive(aligned)
        print(f"  結合後セグメント数: {len(merged)}")

        print("\n  話者統合チェック中...")
        merged, diarization_segments = _merge_similar_speakers(
            merged, diarization_segments, mp3_bytes
        )

        print("\n  声質分析中...")
        voice_features = _analyze_voice(mp3_bytes, merged, diarization_segments)

        return merged, voice_features

    finally:
        Path(tmp_path).unlink(missing_ok=True)
