/*
 * 靜態展示版的 API 墊片。
 *
 * 審查工作台與自檢頁原本透過 /api/... 向 FastAPI 後端取資料；
 * 這份墊片把同樣的請求轉向 data/ 底下由 scripts/build_static_demo.py
 * 從實際 API 擷取的 JSON 快照，讓兩個頁面可以放在任何靜態主機上直接開。
 *
 * 頁面本身的程式碼一行都不改：同一份 dashboard.html / self_check.html，
 * 只是 fetch 被換掉。凡是需要後端即時運算的功能（上傳、新增案件、
 * 自由提問、任意金額的情境模擬）在這裡回傳說明訊息，而不是假裝成功。
 */
(function () {
  const realFetch = window.fetch.bind(window);
  const META_URL = 'data/meta.json';

  function jsonResponse(obj, status) {
    return new Response(JSON.stringify(obj), {
      status: status || 200,
      headers: { 'Content-Type': 'application/json; charset=utf-8' },
    });
  }

  function refuse(msg) {
    return Promise.resolve(jsonResponse({ error: msg }, 400));
  }

  const NOT_IN_SNAPSHOT = '靜態展示版只收錄預設的問題與參數組合；自由輸入需在本機執行完整系統（見 GitHub）。';
  const NO_WRITE = '靜態展示版不支援上傳與新增案件；本機執行完整系統即可使用這個功能（見 GitHub）。';

  // /api/... → data/... 的對應。回傳 null 代表不是 API 請求，原樣放行。
  function mapToSnapshot(url, method) {
    const u = new URL(url, location.href);
    const m = u.pathname.match(/\/api\/(.*)$/);
    if (!m) return null;
    if (method && method.toUpperCase() !== 'GET') return { refuse: NO_WRITE };
    const path = m[1];
    const q = u.searchParams;

    if (path === 'queue') return { file: 'data/queue.json' };
    if (path === 'engagements') return { file: 'data/engagements.json' };

    let mm;
    if ((mm = path.match(/^(overview|crosscheck|cashflow)\/([^/]+)$/)))
      return { file: `data/${mm[2]}/${mm[1]}.json` };
    if ((mm = path.match(/^cases\/([^/]+)\/lookup\/(.+)$/)))
      return { file: `data/${mm[1]}/lookup/${encodeURIComponent(decodeURIComponent(mm[2]))}.json` };
    if (path.startsWith('simulate')) {
      const t = q.get('tenant'), amount = q.get('amount'), days = q.get('days');
      return { file: `data/${t}/simulate/${amount}_${days}.json`, missing: NOT_IN_SNAPSHOT };
    }
    if (path.startsWith('confidence')) {
      const t = q.get('tenant') || 'SHARED', question = q.get('q');
      if (!question) return { file: 'data/confidence_weights.json' };
      return { qa: { tenant: t, question } };
    }
    return { refuse: '靜態展示版沒有這個端點。' };
  }

  async function fetchSnapshot(file, missingMsg) {
    const r = await realFetch(file, { cache: 'no-cache' });
    if (!r.ok) return jsonResponse({ error: missingMsg || NOT_IN_SNAPSHOT }, 404);
    return r;
  }

  // 問答快照用「問題文字」當索引，避免 URL 編碼差異
  const qaIndexCache = {};
  async function qaLookup(tenant, question) {
    if (!qaIndexCache[tenant]) {
      const r = await realFetch(`data/${tenant}/qa/index.json`, { cache: 'no-cache' });
      qaIndexCache[tenant] = r.ok ? await r.json() : {};
    }
    const file = qaIndexCache[tenant][question.trim()];
    if (!file) return jsonResponse({ error: NOT_IN_SNAPSHOT }, 404);
    return fetchSnapshot(`data/${tenant}/qa/${file}`);
  }

  window.fetch = function (input, init) {
    const url = typeof input === 'string' ? input : input.url;
    const method = (init && init.method) || (typeof input !== 'string' && input.method) || 'GET';
    const target = mapToSnapshot(url, method);
    if (!target) return realFetch(input, init);
    if (target.refuse) return refuse(target.refuse);
    if (target.qa) return qaLookup(target.qa.tenant, target.qa.question);
    return fetchSnapshot(target.file, target.missing);
  };

  // ── 畫面上的說明與輸入限制 ────────────────────────────────────────────
  function el(tag, attrs, html) {
    const e = document.createElement(tag);
    Object.entries(attrs || {}).forEach(([k, v]) => e.setAttribute(k, v));
    if (html != null) e.innerHTML = html;
    return e;
  }

  function replaceWithSelect(inputSel, options, current) {
    const inp = document.querySelector(inputSel);
    if (!inp) return;
    const sel = el('select', { id: inp.id, style: inp.getAttribute('style') || '' });
    options.forEach(v => {
      const o = el('option', { value: v }, typeof v === 'number' ? v.toLocaleString('zh-TW') : v);
      if (String(v) === String(current)) o.selected = true;
      sel.appendChild(o);
    });
    inp.replaceWith(sel);
    return sel;
  }

  window.addEventListener('load', async () => {
    let meta = {};
    try { meta = await (await realFetch(META_URL, { cache: 'no-cache' })).json(); } catch (_) {}
    const isSelfCheck = /self-check\.html$/.test(location.pathname);

    const banner = el('div', {
      style: 'background:#fef3c7;color:#78350f;border-bottom:1px solid #f59e0b;'
           + 'padding:8px 16px;font-size:13px;line-height:1.5;',
    }, `<b>線上展示版</b>　畫面與資料為 ${meta.captured_at || ''} 從實際系統擷取的快照`
       + `（示範委任案為合成資料）。上傳、新增案件與自由提問需在本機執行完整系統：`
       + `<a href="${meta.repo || 'https://github.com/wajason/flowmind-ai'}" style="color:#78350f;font-weight:600">GitHub</a>`
       + (isSelfCheck
          ? `　·　<a href="index.html" style="color:#78350f;font-weight:600">銀行端審查工作台 →</a>`
          : `　·　<a href="self-check.html" style="color:#78350f;font-weight:600">中小企業送件前自檢 →</a>`));
    document.body.prepend(banner);

    // 情境模擬：只允許快照裡有的金額 / 天數組合
    if (meta.simulate) {
      replaceWithSelect('#sim-amt', meta.simulate.amounts, meta.simulate.amounts[0]);
      replaceWithSelect('#sim-days', meta.simulate.days, meta.simulate.days[0]);
      const run = document.querySelector('#sim-run');
      if (run) run.onclick = () => window.simulate && window.simulate();
    }

    // 自由提問欄改成提示，預設問題（.ex 連結）照常可點
    const q = document.querySelector('#q');
    if (q) {
      q.placeholder = '線上展示版：請點下方預設問題（自由提問需本機執行）';
    }

    // 自檢頁：案件編號改成下拉，只列快照裡有的委任案
    if (meta.tenants && document.querySelector('#case-id')) {
      const sel = replaceWithSelect('#case-id', meta.tenants, meta.tenants[0]);
      if (sel) sel.onkeydown = e => { if (e.key === 'Enter' && window.runCheck) window.runCheck(); };
    }
  });
})();
