# Podcast Translator 制作ノート

## アーキテクチャ

```
┌──────────────────┐     GitHub API      ┌──────────────────────┐
│  Web UI          │ ──────────────────→ │  GitHub Actions       │
│  (GitHub Pages)  │  workflow_dispatch   │  (ubuntu-latest)      │
│  docs/index.html │ ←────────────────── │  .github/workflows/   │
│                  │  polling + artifact  │  translate.yml        │
└──────────────────┘                     └──────────┬───────────┘
       │                                            │
       │ PAT (localStorage)              ┌──────────▼───────────┐
       │                                 │  Pipeline (Python)    │
       └─ iPhone / PC ブラウザ            │  src/main.py          │
                                         └──────────────────────┘
```

### Web UI → GitHub Actions 方式を選んだ理由

- ユーザーの Mac を常時起動する必要がない
- iPhone からも URL を送信できる
- Cloudflare Worker（API プロキシ）は当初検討したが、アカウントが不要な直接 PAT 方式に変更
- Fine-grained PAT で Actions 権限のみに限定し、セキュリティリスクを最小化

## パイプライン詳細

### STEP 1: 音声ダウンロード + メタデータ収集 (`download.py`)

- **YouTube**: `yt-dlp` で音声を MP3 ダウンロード + `--dump-json` でメタデータ取得 + VTT 字幕取得
- **Podcast RSS**: `feedparser` で最新エピソード解析、`requests` で音声ダウンロード。番組名・著者・説明文を抽出
- **直接 URL**: `requests` でダウンロード（メタデータなし）
- Whisper API の 25MB 上限に対応するため、超過時は 64kbps にダウンコンバートしてトリミング

### STEP 2: メタデータ分析 (`analyze.py`)

- **Claude Haiku** でメタデータ（タイトル・説明文・字幕・チャンネル名）を分析
- 推定する情報: 話者数、各話者の名前・役割（host/guest）・性別
- 話者数のヒントを pyannote に渡すことで過剰分割を抑制
- メタデータが不足している場合はフォールバック（話者数不明、後工程で F0 から推定）

### STEP 3: 話者分離 + 文字起こし (`diarize.py`)

処理の流れ:

1. **pyannote-audio** で話者分離（WAV 16kHz mono に変換して入力）
2. **Whisper API** でタイムスタンプ付き文字起こし（`verbose_json` + `segment` 粒度）
3. **時間重なり照合**: 各 Whisper セグメントと最も時間が重なる話者を割り当て
4. **連続セグメント結合**: 同じ話者の連続発話を1つにマージ
5. **F0 ベース話者統合**: 自己相関法で F0 を推定し、差が 30Hz 未満の話者を同一人物として統合
6. **声質分析**: F0 から pitch / gender_hint / energy を算出

#### F0 推定の技術詳細

- 最初は零交差率（ZCR）を使ったが、1000〜2000Hz という非現実的な値が出た
- **自己相関法**に変更: 30ms 窓で自己相関を計算、70〜400Hz の範囲でピーク検出
- 有声音判定: ピーク強度がゼロラグの 30% 以上
- 外れ値除去: 中央値 ±30% の範囲に収まるフレームのみ採用

#### 話者マッチング（F0 × メタデータ）

pyannote の `Speaker_N` とメタデータの `Speaker_N` は番号が一致しない。
F0 から推定した性別とメタデータの性別情報を照合して正しい対応関係を構築する（`main.py: _match_speakers_by_gender()`）。

```
pyannote Speaker_1 (F0=120Hz → male) → メタデータ "John Smith" (male, host)
pyannote Speaker_2 (F0=240Hz → female) → メタデータ "Jane Doe" (female, guest)
```

### STEP 4: 翻訳 (`translate.py`)

- **Claude Sonnet 4** で翻訳
- 話者プロファイル（性別・役割）を渡し、性別に応じた日本語の口調を指示
  - 男性: 「〜だよね」「〜だと思う」「〜じゃないかな」
  - 女性: 「〜よね」「〜だと思うわ」「〜じゃないかしら」
- 出力は JSON 配列: `[{"speaker": "Speaker_1", "text": "..."}, ...]`
- 1セグメント 200文字以内を目安に分割

### STEP 5: TTS (`tts.py`)

- **Gemini Flash TTS** (`gemini-2.5-flash-preview-tts`)
- 30種のプリセットボイスから F0 帯域とエネルギーに基づいて自動選択
- マルチスピーカーモードは1チャンク最大2人の制約あり → 3人目が現れたらチャンク分割
- 各ボイスに詳細な SPEECH_DIRECTION を付与（声質の安定性向上）
- 読み辞書: 固有名詞のカタカナ読みを Gemini Flash で自動生成し、`input/custom_readings.json` に蓄積

## ファイル構成

```
podcast-translator/
├── .github/workflows/
│   └── translate.yml          # GitHub Actions ワークフロー
├── docs/
│   ├── index.html             # Web UI (GitHub Pages)
│   ├── MANUAL.md              # 運用マニュアル
│   └── DEVELOPMENT.md         # この制作ノート
├── src/
│   ├── main.py                # パイプライン統合
│   ├── download.py            # 音声ダウンロード + メタデータ
│   ├── analyze.py             # メタデータ分析 (Claude Haiku)
│   ├── diarize.py             # 話者分離 + 文字起こし (pyannote + Whisper)
│   ├── translate.py           # 翻訳 (Claude Sonnet)
│   ├── tts.py                 # 音声生成 (Gemini Flash TTS)
│   └── transcribe.py          # (未使用) 単体 Whisper 文字起こし
├── input/
│   └── custom_readings.json   # TTS 読み辞書
├── output/
│   ├── audio/                 # 生成 MP3
│   └── scripts/               # 翻訳スクリプト JSON
├── requirements.txt
├── .env                       # API キー (git 管理外)
└── CLAUDE.md                  # Claude Code 用プロジェクト設定
```

## 使用 API と料金感

| API | モデル | 用途 | 1回あたりの目安 |
|-----|--------|------|----------------|
| Anthropic | Claude Haiku 4.5 | メタデータ分析 | ~$0.001 |
| Anthropic | Claude Sonnet 4 | 翻訳 | ~$0.05 |
| OpenAI | Whisper-1 | 文字起こし | ~$0.06/10分 |
| Google | Gemini Flash TTS | 音声生成 | 無料枠内（~100 req/day） |
| Google | Gemini Flash | 読み辞書生成 | 無料枠内 |
| HuggingFace | pyannote 3.1 | 話者分離 | 無料（ローカル推論） |

10分の動画1本あたり合計 **約 $0.10〜0.15**（Gemini 無料枠利用時）。

## 開発で遭遇した問題と解決

### pyannote-audio v4 の API 変更
- `use_auth_token` → `token` に変更
- 戻り値が `DiarizeOutput` → `getattr(result, "speaker_diarization", result)` で `Annotation` を取得
- WAV 16kHz mono に事前変換しないとサンプル数不整合エラー

### Whisper API の属性アクセス
- `response_format="verbose_json"` の `segments` はオブジェクト → `seg.text` で属性アクセス（`seg["text"]` ではない）

### Gemini TTS のマルチスピーカー制限
- `enabled_voices must equal 2` → 1チャンクに3人以上入るとエラー
- `_build_chunks()` で話者数を監視し、3人目が来たらチャンク分割

### 話者の性別入れ替わり問題
- pyannote の Speaker_N 番号とメタデータの Speaker_N 番号は無関係
- F0 → 性別推定 → メタデータの性別とマッチング → 正しい対応関係を構築
- TTS の voice_features キーも実名にリネームしないと声が入れ替わる

### F0 推定の精度
- 零交差率: 高調波に引きずられて 1000Hz+ の値が出る → 不採用
- 自己相関法: 70-400Hz のラグ範囲でピーク検出、有声音判定付きで正確な F0 を取得

## 姉妹プロジェクト

- `~/trail-podcast/` — トレイルランニング特化の AI Podcast
  - TTS のチャンク分割、SPEECH_DIRECTION、読み辞書の仕組みはここから流用
  - Whisper のトリミング処理も参考
