# Podcast Translator

## 概要
任意の外国語ポッドキャスト/YouTube動画を日本語に翻訳・読み上げるツール。
話者の声質を解析し、Gemini TTS で近い声質で日本語音声を生成する。

## 姉妹プロジェクト
- `~/trail-podcast/` — トレイルランニング特化のAI Podcast（本プロジェクトの原型）
- tts.py、読み辞書、Whisper文字起こし等のコードを参考にできる

## パイプライン
```
URL入力 → 音声取得 → Whisper文字起こし → 話者分離・声質分析 → Claude翻訳 → Gemini TTS → MP3出力
```

## 設計方針

### 入力
- Podcast エピソードの個別URL（音声直リンクまたはエピソードページ）
- YouTube 動画の個別URL
- 1本ずつ処理（バッチではない）

### 文字起こし
- OpenAI Whisper API（言語自動検出、`language` パラメータは指定しない）
- 長い音声は分割して処理（Whisper上限対策: trail-podcast の `_trim_mp3_bytes` 参照）

### 話者分離（Speaker Diarization）
- `pyannote-audio` で話者を分離
- 各話者の声質特徴を抽出（ピッチ帯域、話速、エネルギー等）
- メタデータとして保持: `{"speaker_1": {"pitch": "low", "pace": "slow", ...}, ...}`
- 単一話者の場合はスキップ

### 声質 → Gemini TTS ボイスマッピング
- Gemini TTS のプリセットボイス一覧から最も近いものを自動選択
- SPEECH_DIRECTION で声質特徴（ピッチ・テンポ・トーン）を指示
- trail-podcast の VOICE_CONFIG / SPEECH_DIRECTION パターンを流用

### 翻訳（Claude）
- 原文に忠実な翻訳（意訳・再構成はしない）
- 冗長な部分（繰り返し、フィラー、脱線）は自然にカットしてよい
- 話者ラベルを維持: `[Speaker 1] こんにちは...`
- 出力は JSON: `[{"speaker": "Speaker_1", "text": "..."}, ...]`

### TTS（Gemini Flash TTS）
- trail-podcast の tts.py をベースに汎用化
- チャンクサイズ: 3000文字（trail-podcast で検証済み）
- スピーカー交代時に [pause: 0.8s] 挿入
- 読み辞書の自動更新（Gemini Flash で固有名詞のカタカナ読み生成）

### 出力
- MP3 ファイル（output/ ディレクトリ）
- RSS配信は不要（ローカル利用）

## 将来的な拡張
- Web UI（Streamlit or FastAPI）で URL を入力 → MP3 ダウンロード
- 処理状況のプログレス表示
- 翻訳結果のプレビュー・編集機能

## 技術スタック
- Python 3.12+
- anthropic（Claude API: 翻訳）
- openai（Whisper API: 文字起こし）
- google-genai（Gemini Flash TTS: 音声生成）
- pyannote-audio（話者分離）
- pydub（音声加工）
- yt-dlp（YouTube音声取得）
- feedparser（Podcast RSS解析）

## API キー（.env）
- ANTHROPIC_API_KEY
- OPENAI_API_KEY
- GEMINI_API_KEY
- HF_AUTH_TOKEN（pyannote-audio 用、Hugging Face トークン）

## trail-podcast から流用するコード
- `tts.py`: TTS全般（チャンク分割、Gemini API呼び出し、PCM→AudioSegment、読み辞書）
- `collect.py`: `_whisper_transcribe()`, `_trim_mp3_bytes()`, YouTube音声取得
- `translate.py`: Claude API呼び出しパターン、スクリプトパース
