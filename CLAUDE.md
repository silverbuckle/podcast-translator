# Podcast Translator

## 概要
外国語 Podcast / YouTube 動画の URL → 日本語翻訳 MP3 を生成する CLI ツール。
話者を自動識別し、性別に応じた声質・口調で日本語音声を出力する。

## 実行方法
```bash
cd ~/podcast-translator
source .venv/bin/activate
python src/main.py "URL"
```
出力: `output/audio/<hash>.mp3`

## パイプライン (`src/main.py`)
```
URL → download → analyze_metadata → diarize_and_transcribe → translate → tts → MP3
```

| STEP | ファイル | 使用 API / ツール |
|------|----------|----------|
| 1. 音声DL + メタデータ | `download.py` | yt-dlp (Chrome cookies), feedparser |
| 2. メタデータ分析 | `analyze.py` | Claude Haiku 4.5 |
| 3. 話者分離 + 文字起こし | `diarize.py` | pyannote-audio (MPS), Whisper API |
| 4. 翻訳 | `translate.py` | Claude Sonnet 4 |
| 5. TTS | `tts.py` | Gemini Flash TTS |

## ファイル構成
```
src/main.py              # パイプライン統合
src/download.py          # 音声DL + メタデータ収集
src/analyze.py           # メタデータ分析 (Claude Haiku)
src/diarize.py           # 話者分離 + 文字起こし (pyannote + Whisper)
src/translate.py         # 翻訳 (Claude Sonnet)
src/tts.py               # 音声生成 (Gemini Flash TTS)
docs/MANUAL.md           # 運用マニュアル
docs/DEVELOPMENT.md      # 制作ノート
input/custom_readings.json  # TTS 読み辞書
```

## 技術的な要点

### ローカル実行環境
- Apple Silicon Mac (M4) + MPS で pyannote を高速実行
- Chrome の cookies を直接利用（`--cookies-from-browser chrome`）
- 45分以上の音声は40分チャンクに自動分割（Whisper API の 25MB 制限対応）
- 長い原稿は80行バッチに分割して翻訳

### 話者識別
- pyannote-audio v4 で話者分離（MPS 加速）
- 自己相関法で F0 推定 → 性別判定 → メタデータの性別とマッチング
- F0 差 < 30Hz の話者を統合して過剰分割を修正
- F0 で性別区別不可の場合、発話量順でメタデータにマッチング

### 翻訳
- 話者プロファイル（性別・役割）に応じた日本語の口調（男性言葉/女性言葉）
- 1セグメント 200文字以内で分割
- 長文は80行バッチ × コンテキスト3セグメントで一貫性維持

### TTS
- 30種プリセットボイスから F0 帯域 + エネルギーで自動選択
- マルチスピーカー最大2人/チャンク → 3人目でチャンク分割
- 読み辞書: 固有名詞のカタカナ読みを Gemini Flash で自動生成

## API キー（.env）
- `ANTHROPIC_API_KEY` — Claude API（メタデータ分析 + 翻訳）
- `OPENAI_API_KEY` — Whisper API（文字起こし）
- `GEMINI_API_KEY` — Gemini TTS（音声生成）
- `HF_AUTH_TOKEN` — pyannote-audio（HuggingFace、話者分離）

## 姉妹プロジェクト
- `~/trail-podcast/` — トレイルランニング特化 AI Podcast（TTS・読み辞書の原型）

## 将来計画
- Mac mini 常時稼働サーバー化（FastAPI Web UI、iPhone からアクセス）
- Whisper ローカル実行（コスト削減・長時間対応）
