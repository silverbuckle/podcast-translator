# diarize.py - 話者分離 + Whisper タイムスタンプ照合

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


def _diarize(audio_path: str, min_speakers: int | None = None,
             max_speakers: int | None = None) -> list[dict]:
    """pyannote-audio で話者分離する。

    Args:
        audio_path: WAV ファイルパス
        min_speakers: 最小話者数（指定すると過剰分割を抑制）
        max_speakers: 最大話者数

    Returns:
        [{"speaker": "SPEAKER_00", "start": 0.5, "end": 3.2}, ...]
    """
    hf_token = os.getenv("HF_AUTH_TOKEN")
    if not hf_token:
        raise ValueError("HF_AUTH_TOKEN が .env に設定されていません（pyannote-audio 用）")

    print("  pyannote-audio 話者分離中...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )

    # GPU があれば使う
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        pipeline.to(torch.device("mps"))

    # 話者数のヒントを渡す
    kwargs = {}
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    result = pipeline(audio_path, **kwargs)
    # pyannote v4: DiarizeOutput → .speaker_diarization で Annotation を取得
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
    """Whisper API でタイムスタンプ付き文字起こしする。

    Returns:
        [{"text": "hello", "start": 0.0, "end": 1.5}, ...]
    """
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
    """Whisper のテキストセグメントに話者ラベルを付与する。

    各 Whisper セグメントの時間範囲と最も重なりが大きい話者を割り当てる。
    """
    result = []
    for ws in whisper_segments:
        ws_start = ws["start"]
        ws_end = ws["end"]
        ws_duration = ws_end - ws_start

        if ws_duration <= 0:
            continue

        # 各話者との重なり時間を計算
        speaker_overlap: dict[str, float] = {}
        for ds in diarization_segments:
            overlap_start = max(ws_start, ds["start"])
            overlap_end = min(ws_end, ds["end"])
            overlap = max(0, overlap_end - overlap_start)
            if overlap > 0:
                speaker = ds["speaker"]
                speaker_overlap[speaker] = speaker_overlap.get(speaker, 0) + overlap

        # 最も重なりが大きい話者を選択
        if speaker_overlap:
            speaker = max(speaker_overlap, key=speaker_overlap.get)
        else:
            # 重なりがない場合は直近の話者を使う
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
    """自己相関法で基本周波数 (F0) を推定する。

    ゼロ交差率よりもはるかに正確。人間の声の範囲 (70-400Hz) を対象とする。
    """
    # 分析窓（30ms ずつ、音声全体からサンプリング）
    window_size = int(0.03 * sample_rate)
    hop_size = int(0.01 * sample_rate)
    min_lag = int(sample_rate / 400)   # 400Hz → 最小ラグ
    max_lag = int(sample_rate / 70)    # 70Hz → 最大ラグ

    f0_values = []

    for start in range(0, len(samples) - window_size, hop_size):
        frame = samples[start:start + window_size]

        # 無音フレームをスキップ
        if np.max(np.abs(frame)) < np.max(np.abs(samples)) * 0.05:
            continue

        # 自己相関を計算
        frame = frame - np.mean(frame)
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr) // 2:]  # 正のラグのみ

        if max_lag >= len(corr):
            continue

        # 人間の声の範囲内でピークを探す
        search_region = corr[min_lag:max_lag]
        if len(search_region) == 0:
            continue

        peak_idx = np.argmax(search_region) + min_lag

        # ピークが十分に強い場合のみ採用（有声音判定）
        if corr[peak_idx] > 0.3 * corr[0]:
            f0 = sample_rate / peak_idx
            f0_values.append(f0)

    if not f0_values:
        return 0.0

    # 外れ値を除去して中央値を返す
    f0_arr = np.array(f0_values)
    median = np.median(f0_arr)
    filtered = f0_arr[np.abs(f0_arr - median) < median * 0.3]
    return float(np.median(filtered)) if len(filtered) > 0 else float(median)


def _analyze_voice(mp3_bytes: bytes, segments: list[dict],
                   diarization_segments: list[dict]) -> dict[str, dict]:
    """各話者の声質特徴を分析する（自己相関法による F0 推定）。

    Returns:
        {"Speaker_1": {"pitch": "low", "energy": "high", "gender_hint": "male"}, ...}
    """
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

            # 自己相関法で F0 を推定
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
    """F0 が近い話者を統合して過剰分割を修正する。

    F0 の差が 30Hz 以内の話者は同一人物とみなす。
    """
    audio = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    sample_rate = audio.frame_rate

    speakers = sorted(set(s["speaker"] for s in diarization_segments))
    if len(speakers) <= 2:
        return segments, diarization_segments

    # 各話者の F0 を計算
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

    # F0 が近い話者をグループ化
    merge_map: dict[str, str] = {}  # old_speaker → canonical_speaker
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

    # 統合不要なら何もしない
    if len(set(merge_map.values())) == len(speakers):
        return segments, diarization_segments

    # セグメントのラベルを更新
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

    # 話者名を振り直す
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


def diarize_and_transcribe(mp3_bytes: bytes,
                          min_speakers: int | None = None,
                          max_speakers: int | None = None,
                          ) -> tuple[list[dict], dict[str, dict]]:
    """話者分離 + 文字起こし + 照合を行う。

    Args:
        mp3_bytes: MP3 音声データ
        min_speakers: 最小話者数（メタデータ分析結果から）
        max_speakers: 最大話者数

    Returns:
        (segments, voice_features)
        segments: [{"speaker": "Speaker_1", "text": "...", "start": 0.0, "end": 1.5}, ...]
        voice_features: {"Speaker_1": {"pitch": "low", "gender_hint": "male", ...}, ...}
    """
    if min_speakers:
        print(f"  話者数ヒント: {min_speakers}人")

    # pyannote 用に WAV 16kHz mono に変換（サンプル数の不整合を防ぐ）
    audio = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
    audio = audio.set_frame_rate(16000).set_channels(1)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        audio.export(f, format="wav")
        tmp_path = f.name

    try:
        # 1. 話者分離（話者数ヒントを渡す）
        diarization_segments = _diarize(
            tmp_path, min_speakers=min_speakers, max_speakers=max_speakers
        )

        # 話者が1人の場合
        if len(set(s["speaker"] for s in diarization_segments)) <= 1:
            print("  話者が1人のみ → 話者分離スキップ")
            whisper_segments = _whisper_with_timestamps(mp3_bytes)
            segments = [{"speaker": "Narrator", "text": s["text"],
                         "start": s["start"], "end": s["end"]}
                        for s in whisper_segments]
            return _merge_consecutive(segments), {}

        # 2. Whisper タイムスタンプ付き文字起こし
        whisper_segments = _whisper_with_timestamps(mp3_bytes)

        # 3. 話者ラベルの照合
        print("\n  話者ラベル照合中...")
        aligned = _align_speakers(whisper_segments, diarization_segments)

        # 4. 同一話者の連続セグメントを結合
        merged = _merge_consecutive(aligned)
        print(f"  結合後セグメント数: {len(merged)}")

        # 5. F0 ベースで過剰分割された話者を統合
        print("\n  話者統合チェック中...")
        merged, diarization_segments = _merge_similar_speakers(
            merged, diarization_segments, mp3_bytes
        )

        # 6. 声質分析
        print("\n  声質分析中...")
        voice_features = _analyze_voice(mp3_bytes, merged, diarization_segments)

        return merged, voice_features

    finally:
        Path(tmp_path).unlink(missing_ok=True)
