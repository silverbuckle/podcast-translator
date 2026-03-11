# main.py - パイプライン全体を実行

import sys
import json
import hashlib
from pathlib import Path

from download import download
from analyze import analyze_metadata
from diarize import diarize_and_transcribe
from translate import translate
from tts import tts

SCRIPTS_DIR = Path(__file__).parent.parent / "output" / "scripts"


def _make_output_name(url: str) -> str:
    """URL からファイル名を生成する。"""
    return hashlib.md5(url.encode()).hexdigest()[:12]


def _match_speakers_by_gender(voice_features: dict[str, dict],
                              speaker_info: list[dict]) -> dict[str, dict]:
    """F0 の性別推定とメタデータの性別情報を照合し、正しい対応を返す。

    pyannote の Speaker_N とメタデータの Speaker_N は番号が一致するとは限らない。
    F0 から推定した gender_hint とメタデータの gender をマッチングして
    正しい対応関係を構築する。

    Returns:
        {pyannote_speaker_id: metadata_speaker_dict, ...}
    """
    if not voice_features or not speaker_info:
        return {}

    # メタデータの話者を性別でグループ化
    meta_male = [sp for sp in speaker_info if sp.get("gender") == "male"]
    meta_female = [sp for sp in speaker_info if sp.get("gender") == "female"]
    meta_unknown = [sp for sp in speaker_info if sp.get("gender") not in ("male", "female")]

    # pyannote の話者を F0 でソート（低い=男性の可能性が高い）
    sorted_speakers = sorted(voice_features.items(),
                             key=lambda x: x[1].get("estimated_f0_hz", 200))

    mapping: dict[str, dict] = {}
    used_meta: set[str] = set()

    # まず性別が明確なものをマッチング
    for speaker_id, features in sorted_speakers:
        f0_gender = features.get("gender_hint", "unknown")

        best_match = None
        if f0_gender == "male" and meta_male:
            # 未使用の男性メタデータから選択
            for sp in meta_male:
                if sp.get("id", "") not in used_meta:
                    best_match = sp
                    break
        elif f0_gender == "female" and meta_female:
            for sp in meta_female:
                if sp.get("id", "") not in used_meta:
                    best_match = sp
                    break

        if best_match:
            mapping[speaker_id] = best_match
            used_meta.add(best_match.get("id", ""))

    # 未マッチの話者を残りのメタデータに割り当て
    remaining_meta = [sp for sp in speaker_info if sp.get("id", "") not in used_meta]
    for speaker_id, _ in sorted_speakers:
        if speaker_id not in mapping and remaining_meta:
            mapping[speaker_id] = remaining_meta.pop(0)

    return mapping


def _rename_speakers(segments: list[dict], name_map: dict[str, str]) -> list[dict]:
    """話者ラベルを実名に置換する。"""
    if not name_map:
        return segments

    renamed = []
    for seg in segments:
        new_seg = seg.copy()
        if seg["speaker"] in name_map:
            new_seg["speaker"] = name_map[seg["speaker"]]
        renamed.append(new_seg)

    return renamed


def run(url: str) -> Path:
    """URL → 日本語 MP3 の全パイプラインを実行する。"""
    output_name = _make_output_name(url)

    # 1. 音声ダウンロード + メタデータ収集
    print("=" * 60)
    print("STEP 1: 音声ダウンロード + メタデータ収集")
    print("=" * 60)
    mp3_bytes, metadata = download(url)
    print(f"  タイトル: {metadata.get('title', '?')}")
    print(f"  サイズ: {len(mp3_bytes) / 1024 / 1024:.1f} MB")

    # 2. メタデータ分析（出演者推定）
    print("\n" + "=" * 60)
    print("STEP 2: メタデータ分析 (Claude)")
    print("=" * 60)
    analysis = analyze_metadata(metadata)
    num_speakers = analysis.get("num_speakers")
    speaker_info = analysis.get("speakers", [])

    # 3. 話者分離 + 文字起こし
    print("\n" + "=" * 60)
    print("STEP 3: 話者分離 + 文字起こし (pyannote + Whisper)")
    print("=" * 60)
    segments, voice_features = diarize_and_transcribe(
        mp3_bytes,
        min_speakers=num_speakers if num_speakers else None,
        max_speakers=num_speakers if num_speakers else None,
    )

    # F0 の性別推定とメタデータの性別を照合して正しい対応関係を構築
    speaker_mapping = _match_speakers_by_gender(voice_features, speaker_info)
    name_map: dict[str, str] = {}
    matched_profiles: list[dict] = []

    if speaker_mapping:
        print("\n  話者マッチング (F0 × メタデータ):")
        for pyannote_id, meta_sp in speaker_mapping.items():
            f0 = voice_features.get(pyannote_id, {}).get("estimated_f0_hz", "?")
            meta_name = meta_sp.get("name", "Unknown")
            meta_gender = meta_sp.get("gender", "?")
            print(f"    {pyannote_id} (F0={f0}Hz) → {meta_name} ({meta_gender})")
            if meta_name != "Unknown":
                name_map[pyannote_id] = meta_name
            # voice_features の性別もメタデータに合わせて補正
            if meta_gender in ("male", "female"):
                voice_features[pyannote_id]["gender_hint"] = meta_gender
            # マッチしたプロファイルを pyannote の Speaker_N で再ラベル
            matched_profiles.append({**meta_sp, "id": pyannote_id})

    # 結果を表示
    for seg in segments:
        display_name = name_map.get(seg["speaker"], seg["speaker"])
        preview = seg["text"][:60]
        print(f"  [{display_name}] {preview}...")

    # 4. 翻訳
    print("\n" + "=" * 60)
    print("STEP 4: 翻訳 (Claude)")
    print("=" * 60)
    labeled_transcript = "\n".join(
        f"[{seg['speaker']}] {seg['text']}" for seg in segments
    )

    # コンテキスト情報と話者プロファイルを翻訳に渡す
    context = analysis.get("context", "")
    translated_segments = translate(
        labeled_transcript,
        speakers=voice_features,
        context=context,
        speaker_profiles=matched_profiles if matched_profiles else speaker_info,
    )

    # 話者名を実名に置換
    translated_segments = _rename_speakers(translated_segments, name_map)

    # voice_features のキーも実名に揃える（TTS のボイス割り当てで使うため）
    if name_map:
        voice_features = {
            name_map.get(k, k): v for k, v in voice_features.items()
        }

    # 翻訳結果を保存
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    script_path = SCRIPTS_DIR / f"{output_name}.json"
    with open(script_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": metadata,
            "analysis": analysis,
            "voice_features": voice_features,
            "segments": translated_segments,
        }, f, ensure_ascii=False, indent=2)
    print(f"  スクリプト保存: {script_path}")

    # 5. TTS
    print("\n" + "=" * 60)
    print("STEP 5: 音声生成 (Gemini TTS)")
    print("=" * 60)
    mp3_path = tts(translated_segments, output_name, voice_features=voice_features)

    print("\n" + "=" * 60)
    print("完了!")
    print(f"  タイトル: {metadata.get('title', '?')}")
    print(f"  出力: {mp3_path}")
    print("=" * 60)

    return mp3_path



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使い方: python src/main.py <URL>")
        print("  例: python src/main.py https://www.youtube.com/watch?v=xxxxx")
        sys.exit(1)

    url = sys.argv[1]
    run(url)
