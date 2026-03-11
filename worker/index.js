// Cloudflare Worker - GitHub Actions API プロキシ
// GitHub PAT を隠蔽しつつ、workflow_dispatch をトリガーする

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    };

    // CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }

    // POST /api/trigger - ワークフロー起動
    if (url.pathname === '/api/trigger' && request.method === 'POST') {
      try {
        const { url: podcastUrl } = await request.json();
        if (!podcastUrl) {
          return jsonResponse({ error: 'url is required' }, 400, corsHeaders);
        }

        // GitHub Actions workflow_dispatch をトリガー
        const ghResp = await fetch(
          `https://api.github.com/repos/${env.GITHUB_REPO}/actions/workflows/translate.yml/dispatches`,
          {
            method: 'POST',
            headers: {
              'Authorization': `Bearer ${env.GITHUB_PAT}`,
              'Accept': 'application/vnd.github.v3+json',
              'User-Agent': 'podcast-translator-worker',
            },
            body: JSON.stringify({
              ref: 'main',
              inputs: { url: podcastUrl },
            }),
          }
        );

        if (!ghResp.ok) {
          const text = await ghResp.text();
          return jsonResponse({ error: `GitHub API error: ${ghResp.status}`, detail: text }, 502, corsHeaders);
        }

        // workflow_dispatch は run_id を返さないので、最新の run を取得
        await new Promise(r => setTimeout(r, 2000));
        const runsResp = await fetch(
          `https://api.github.com/repos/${env.GITHUB_REPO}/actions/workflows/translate.yml/runs?per_page=1`,
          {
            headers: {
              'Authorization': `Bearer ${env.GITHUB_PAT}`,
              'Accept': 'application/vnd.github.v3+json',
              'User-Agent': 'podcast-translator-worker',
            },
          }
        );
        const runsData = await runsResp.json();
        const runId = runsData.workflow_runs?.[0]?.id;

        return jsonResponse({ ok: true, run_id: runId }, 200, corsHeaders);

      } catch (e) {
        return jsonResponse({ error: e.message }, 500, corsHeaders);
      }
    }

    // GET /api/status/:runId - ステータス確認
    if (url.pathname.startsWith('/api/status/') && request.method === 'GET') {
      const runId = url.pathname.split('/').pop();
      try {
        const ghResp = await fetch(
          `https://api.github.com/repos/${env.GITHUB_REPO}/actions/runs/${runId}`,
          {
            headers: {
              'Authorization': `Bearer ${env.GITHUB_PAT}`,
              'Accept': 'application/vnd.github.v3+json',
              'User-Agent': 'podcast-translator-worker',
            },
          }
        );
        const data = await ghResp.json();

        const result = {
          status: data.status === 'completed' ? data.conclusion : data.status,
          run_id: runId,
        };

        // 完了時はアーティファクトURLを取得
        if (data.conclusion === 'success') {
          const artResp = await fetch(
            `https://api.github.com/repos/${env.GITHUB_REPO}/actions/runs/${runId}/artifacts`,
            {
              headers: {
                'Authorization': `Bearer ${env.GITHUB_PAT}`,
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'podcast-translator-worker',
              },
            }
          );
          const artData = await artResp.json();
          const audio = artData.artifacts?.find(a => a.name === 'translated-audio');
          if (audio) {
            result.artifact_url = audio.archive_download_url;
          }
        }

        return jsonResponse(result, 200, corsHeaders);

      } catch (e) {
        return jsonResponse({ error: e.message }, 500, corsHeaders);
      }
    }

    return jsonResponse({ error: 'Not found' }, 404, corsHeaders);
  },
};

function jsonResponse(data, status, headers = {}) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json', ...headers },
  });
}
