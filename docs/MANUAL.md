# Podcast Translator 運用マニュアル

## 概要

外国語の Podcast / YouTube 動画の URL を入力すると、日本語に翻訳された MP3 音声を生成するツール。
話者を自動識別し、性別に応じた声質・口調で日本語音声を出力する。

## 使い方

### Web UI（iPhone / PC）

1. https://silverbuckle.github.io/podcast-translator/ にアクセス
2. 初回のみ GitHub Personal Access Token を入力して「保存」
3. Podcast / YouTube の URL を貼り付けて「翻訳」
4. 処理完了後、MP3 ダウンロードリンクが表示される

**iPhone ショートカット**: ホーム画面に追加すれば、Safari の共有メニューから直接 URL を送れる（`?url=` パラメータ対応）。

### CLI（ローカル実行）

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

## GitHub PAT の作成方法

1. https://github.com/settings/personal-access-tokens/new にアクセス
2. **Token name**: `podcast-translator`
3. **Expiration**: 90 days（任意）
4. **Repository access**: `Only select repositories` → `podcast-translator`
5. **Permissions**: `Actions` → `Read and write`
6. **Generate token** をクリック
7. Web UI の設定欄に貼り付けて保存（localStorage に保管される）

## 必要な API キー

GitHub Actions の Secrets（Settings → Secrets and variables → Actions）に設定済み：

| Secret 名 | 用途 | 取得先 |
|-----------|------|--------|
| `ANTHROPIC_API_KEY` | Claude API（メタデータ分析・翻訳） | https://console.anthropic.com/ |
| `OPENAI_API_KEY` | Whisper API（文字起こし） | https://platform.openai.com/ |
| `GEMINI_API_KEY` | Gemini Flash TTS（音声生成） | https://aistudio.google.com/ |
| `HF_AUTH_TOKEN` | pyannote-audio（話者分離） | https://huggingface.co/settings/tokens |

ローカル実行時は `.env` ファイルに同じキーを設定する。

## 処理時間の目安

10分程度の音声で約5〜10分（GitHub Actions）。大部分は話者分離（pyannote）と TTS（Gemini）にかかる。

## 制限事項

- **音声の長さ**: Whisper API の上限により約50分（24MB / 64kbps 換算）まで。超過分は自動トリミング
- **Gemini TTS**: 無料枠は約 100 requests/day。長い音声はチャンク数が多くなるため注意
- **話者数**: 2〜3人の会話に最適化。4人以上は精度が下がる可能性あり
- **GitHub Actions**: workflow_dispatch のためリポジトリは public（コードに秘密情報は含まない）

## トラブルシューティング

### Web UI で「GitHub API 401」が出る
→ PAT の有効期限切れ。新しい PAT を作成して再入力。

### 翻訳で話者の性別が逆になる
→ メタデータ不足時に F0（声の高さ）から性別を推定するため、声質によっては誤判定する。YouTube 動画であれば字幕や説明文が手がかりになる。

### TTS で声が入れ替わる
→ Gemini TTS のマルチスピーカーモードは2人まで。3人以上の場合はチャンク分割で対応するが、チャンク境界で不安定になることがある。

### GitHub Actions がタイムアウトする
→ デフォルトは60分。非常に長い音声の場合は `translate.yml` の `timeout-minutes` を調整。
