# Podcast Translator 運用マニュアル

## 概要

外国語の Podcast / YouTube 動画の URL を入力すると、日本語に翻訳された MP3 音声を生成する CLI ツール。
話者を自動識別し、性別に応じた声質・口調で日本語音声を出力する。

## 使い方

```bash
cd ~/podcast-translator
source .venv/bin/activate
python src/main.py "https://www.youtube.com/watch?v=xxxxx"
```

出力先: `output/audio/<hash>.mp3`

## 対応 URL

| 種類 | 例 | 備考 |
|------|------|------|
| YouTube | `https://www.youtube.com/watch?v=...` | メタデータ・字幕も自動取得 |
| YouTube Short | `https://youtu.be/...` | 同上 |
| Podcast RSS | `https://feeds.example.com/podcast.xml` | 最新エピソードを処理 |
| 直接音声 | `https://example.com/episode.mp3` | メタデータなし |

## 必要な API キー

`.env` ファイルに設定:

| キー名 | 用途 | 取得先 |
|--------|------|--------|
| `ANTHROPIC_API_KEY` | Claude API（メタデータ分析・翻訳） | https://console.anthropic.com/ |
| `OPENAI_API_KEY` | Whisper API（文字起こし） | https://platform.openai.com/ |
| `GEMINI_API_KEY` | Gemini Flash TTS（音声生成） | https://aistudio.google.com/ |
| `HF_AUTH_TOKEN` | pyannote-audio（話者分離） | https://huggingface.co/settings/tokens |

## 処理時間の目安（M4 Mac）

| 音声の長さ | 処理時間 | 備考 |
|-----------|---------|------|
| 10分 | 5〜10分 | pyannote + TTS が大部分 |
| 30分 | 15〜25分 | |
| 60分 | 30〜50分 | 自動チャンク分割 |
| 90分 | 45〜75分 | |

pyannote（話者分離）は MPS (Apple Silicon GPU) で高速処理される。

## 制限事項

- **Whisper API**: 25MB/回。45分以上の音声は40分チャンクに自動分割して対応
- **Gemini TTS**: 無料枠は約 100 requests/day。60分の音声で 40〜60 チャンク消費
- **話者数**: 2〜3人の会話に最適化。4人以上は精度が下がる可能性あり
- **YouTube**: Chrome の cookies を直接利用（`--cookies-from-browser chrome`）。Chrome が起動中の場合はエラーになることがある → Chrome を閉じて再実行

## トラブルシューティング

### 翻訳で話者の性別が逆になる
→ メタデータ不足時に F0（声の高さ）から性別を推定するため、声質によっては誤判定する。YouTube 動画であれば字幕や説明文が手がかりになる。

### TTS で声が入れ替わる
→ Gemini TTS のマルチスピーカーモードは2人まで。3人以上の場合はチャンク分割で対応するが、チャンク境界で不安定になることがある。

### yt-dlp で YouTube ダウンロードが失敗する
→ `yt-dlp` を最新版に更新: `pip install -U yt-dlp`
→ Chrome を閉じてから再実行（cookies アクセスのため）

### pyannote が遅い
→ MPS が有効か確認: `python -c "import torch; print(torch.backends.mps.is_available())"`
→ `True` が出れば GPU 加速が有効
