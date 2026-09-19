#!/usr/bin/env python
"""
build_static_demo.py — 把審查工作台與自檢頁打包成可放在靜態主機上的展示版。

    python scripts/build_static_demo.py            # 全部（含問答，需 Ollama）
    python scripts/build_static_demo.py --no-qa    # 跳過問答快照

做法：在行程內啟動 flowmind.dashboard 的 FastAPI app（TestClient，不開 port），
逐一呼叫頁面會用到的 GET 端點，把回應原封不動存成 demo/data/ 底下的 JSON；
再把兩份 HTML 複製到 demo/ 並掛上 static_shim.js，讓頁面的 fetch 轉向這些快照。

頁面程式碼不改、數字不加工——展示版看到的就是實際系統在擷取當下算出來的東西。
快照時間寫進 demo/data/meta.json，畫面頂端會顯示。

需要：資料庫已啟動（docker compose up -d）、示範委任案已入庫。
問答快照需要 Ollama 與 gemma4:26b；用 --no-qa 可跳過。
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEMO = ROOT / "demo"
DATA = DEMO / "data"
STATIC = ROOT / "flowmind" / "static"

# 情境模擬允許的參數組合（畫面上會變成下拉選單）
SIM_AMOUNTS = [3_000_000, 5_000_000, 10_000_000, 20_000_000]
SIM_DAYS = [10, 20, 30, 60]

# 問答快照：與 dashboard.html 上「試試：」列出的預設問題一致
QA_QUESTIONS = [
    "跨文件勾稽有一個沒通過是什麼地方？",
    "最大買方占營收多少？逾期狀況如何？",
    "信保基金供應商融資的保證成數最高幾成？",
    "日本的保證成數是多少？",
]


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def _collect_refs(obj, out: set) -> None:
    """crosscheck / overview 回應裡所有 refs 欄位的憑證編號（去掉括號附註）。"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "refs" and isinstance(v, list):
                out.update(str(r).split("(")[0].strip() for r in v)
            _collect_refs(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_refs(v, out)


def _inject(html: str) -> str:
    """掛上墊片；其餘不動。"""
    shim = '<script src="static_shim.js"></script>\n'
    idx = html.find("<script>")
    assert idx > 0, "找不到頁面的 <script> 區塊"
    html = html[:idx] + shim + html[idx:]
    # 自檢頁的 PDF 下載是 <a href>，不走 fetch，直接改成靜態檔路徑
    html = html.replace("href=\"/api/report/${encodeURIComponent(tenant)}\"",
                        "href=\"report/${encodeURIComponent(tenant)}.pdf\"")
    html = html.replace("<title>", "<title>[展示] ", 1)
    return html


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-qa", action="store_true", help="跳過問答快照（不需 Ollama）")
    ap.add_argument("--tenants", nargs="*", help="只擷取這些委任案（預設：佇列裡全部）")
    args = ap.parse_args()

    from fastapi.testclient import TestClient  # noqa: PLC0415
    from flowmind.dashboard import app          # noqa: PLC0415
    c = TestClient(app)

    def get(path: str):
        r = c.get(path)
        if r.status_code != 200:
            raise SystemExit(f"{path} → {r.status_code}: {r.text[:200]}")
        return r.json()

    # --no-qa 時保留既有的問答快照（那是唯一需要 LLM 的部分，重抓要好幾分鐘）
    kept_qa: dict[str, Path] = {}
    if DATA.exists():
        if args.no_qa:
            import tempfile                                # noqa: PLC0415
            stash = Path(tempfile.mkdtemp(prefix="flowmind_qa_"))
            for qa in DATA.glob("*/qa"):
                shutil.copytree(qa, stash / qa.parent.name)
                kept_qa[qa.parent.name] = stash / qa.parent.name
        shutil.rmtree(DATA)
    (DEMO / "report").mkdir(parents=True, exist_ok=True)

    queue = get("/api/queue")
    _dump(DATA / "queue.json", queue)
    _dump(DATA / "engagements.json", get("/api/engagements"))
    _dump(DATA / "confidence_weights.json", get("/api/confidence"))

    tenants = args.tenants or [row["tenant_id"] for row in queue]
    print(f"委任案：{tenants}")

    for t in tenants:
        print(f"── {t}")
        refs: set[str] = set()
        for name in ("overview", "crosscheck", "cashflow"):
            obj = get(f"/api/{name}/{t}")
            _dump(DATA / t / f"{name}.json", obj)
            _collect_refs(obj, refs)
        print(f"   證據回查 {len(refs)} 筆")
        for ref in sorted(refs):
            r = c.get(f"/api/cases/{t}/lookup/{quote(ref, safe='')}")
            if r.status_code == 200:
                _dump(DATA / t / "lookup" / f"{quote(ref, safe='')}.json", r.json())
        for a in SIM_AMOUNTS:
            for d in SIM_DAYS:
                _dump(DATA / t / "simulate" / f"{a}_{d}.json",
                      get(f"/api/simulate?tenant={t}&amount={a}&days={d}"))
        # PDF 報告
        r = c.get(f"/api/report/{t}")
        if r.status_code == 200:
            (DEMO / "report" / f"{t}.pdf").write_bytes(r.content)
            print("   報告 PDF ✓")
        if args.no_qa and t in kept_qa:
            shutil.copytree(kept_qa[t], DATA / t / "qa")
            print("   問答快照沿用既有")
        if not args.no_qa:
            index = {}
            for i, q in enumerate(QA_QUESTIONS, 1):
                print(f"   問答 {i}/{len(QA_QUESTIONS)}：{q}", flush=True)
                obj = get(f"/api/confidence?tenant={t}&q={quote(q)}")
                fname = f"q{i}.json"
                _dump(DATA / t / "qa" / fname, obj)
                index[q] = fname
            _dump(DATA / t / "qa" / "index.json", index)

    tz = timezone(timedelta(hours=8))
    meta = {
        "captured_at": datetime.now(tz).strftime("%Y-%m-%d %H:%M (UTC+8)"),
        "repo": "https://github.com/wajason/flowmind-ai",
        "tenants": tenants,
        "simulate": {"amounts": SIM_AMOUNTS, "days": SIM_DAYS},
        "qa_questions": QA_QUESTIONS,
    }
    _dump(DATA / "meta.json", meta)

    dash = (STATIC / "dashboard.html").read_text(encoding="utf-8")
    selfc = (STATIC / "self_check.html").read_text(encoding="utf-8")
    (DEMO / "index.html").write_text(_inject(dash), encoding="utf-8")
    (DEMO / "self-check.html").write_text(_inject(selfc), encoding="utf-8")
    (DEMO / ".nojekyll").write_text("", encoding="utf-8")

    n = sum(1 for _ in DATA.rglob("*.json"))
    print(f"完成：{n} 個快照 → {DEMO.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
