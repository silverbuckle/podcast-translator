# Podcast Translator

## 概要
外国語 Podcast / YouTube 動画の URL → 日本語翻訳 MP3 を生成するツール。
話者を自動識別し、性別に応じた声質・口調で日本語音声を出力する。

## アーキテクチャ
```
iPhone/PC ブラウザ → Web UI (GitHub Pages) → GitHub Actions → MP3 artifact
```
- Web UI: `docs/index.html`（GitHub PAT を localStorage に保存、GitHub API を直接呼び出し）
- GitHub Actions: `workflow_dispatch` で起動、`translate.yml` でパイプライン実行
- 出力: artifact として MP3 + スクリプト JSON（30日保持）

## パイプライン (`src/main.py`)
```
URL → download → analyze_metadata → diarize_and_transcribe → translate → tts → MP3
```

| STEP | ファイル | 使用 API |
|------|----------|----------|
| 1. 音声DL + メタデータ | `download.py` | yt-dlp, feedparser |
| 2. メタデータ分析 | `analyze.py` | Claude Haiku 4.5 |
| 3. 話者分離 + 文字起こし | `diarize.py` | pyannote-audio, Whisper |
| 4. 翻訳 | `translate.py` | Claude Sonnet 4 |
| 5. TTS | `tts.py` | Gemini Flash TTS |

## ファイル構成
```
docs/index.html          # Web UI (GitHub Pages)
docs/MANUAL.md           # 運用マニュアル
docs/DEVELOPMENT.md      # 制作ノート
src/main.py              # パイプライン統合
src/download.py          # 音声DL + メタデータ収集
src/analyze.py           # メタデータ分析 (Claude Haiku)
src/diarize.py           # 話者分離 + 文字起こし (pyannote + Whisper)
src/translate.py         # 翻訳 (Claude Sonnet)
src/tts.py               # 音声生成 (Gemini Flash TTS)
.github/workflows/translate.yml  # GitHub Actions
input/custom_readings.json       # TTS 読み辞書
```

## 技術的な要点

### 話者識別
- pyannote-audio v4 で話者分離（`token=` で認証、`DiarizeOutput.speaker_diarization` で Annotation 取得）
- 自己相関法で F0 推定 → 性別判定 → メタデータの性別とマッチング（`_match_speakers_by_gender()`）
- F0 差 < 30Hz の話者を統合して過剰分割を修正

### 翻訳
- 話者プロファイル（性別・役割）に応じた日本語の口調（男性言葉/女性言葉）
- 1セグメント 200文字以内で分割

### TTS
- 30種プリセットボイスから F0 帯域 + エネルギーで自動選択
- マルチスピーカー最大2人/チャンク → 3人目でチャンク分割
- 読み辞書: 固有名詞のカタカナ読みを Gemini Flash で自動生成

## API キー
`.env`（ローカル）/ GitHub Secrets（Actions）:
- `ANTHROPIC_API_KEY` — Claude API
- `OPENAI_API_KEY` — Whisper API
- `GEMINI_API_KEY` — Gemini TTS
- `HF_AUTH_TOKEN` — pyannote-audio (HuggingFace)

## 姉妹プロジェクト
- `~/trail-podcast/` — トレイルランニング特化 AI Podcast（TTS・読み辞書の原型）
