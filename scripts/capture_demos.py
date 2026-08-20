#!/usr/bin/env python3
"""
capture_demos.py — 把所有 demo 實際跑一遍並存檔
=============================================================================
產出 out/DEMO_RESULTS.md：每一項 demo 的**真實終端機輸出**，含執行時間與時間戳。

為什麼要有這支程式：
  簡報上放示意圖，任何人都做得出來。
  放真實輸出，代表這套系統當下真的跑得起來。

  而且這份檔案是可重跑的 —— 評審或合作對象拿到原始碼後，
  跑一次就能得到同樣的結果。「可重現」比「好看」重要。

  如果某一項 demo 失敗了，這支程式會**如實記錄失敗訊息**，
  不會跳過。一份只記錄成功項目的報告，跟沒有報告一樣。

Usage:
    python scripts/capture_demos.py
    python scripts/capture_demos.py --only 3 4 5      # 只跑指定編號
    python scripts/capture_demos.py --skip-slow       # 略過需要 LLM 的項目
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
if not Path(PY).exists():
    PY = sys.executable

# 刻意放 docs/ 而不是 out/：out/ 在 .gitignore 裡，
# 但這份是要給同仁與評審看的交付物，必須進版控 ——
# 而且它進版控之後，git diff 就能看出「這次改動讓哪一項 demo 的輸出變了」。
OUT = ROOT / "docs" / "DEMO_RESULTS.md"

# slow=True 代表需要呼叫 LLM，在 8GB VRAM 上每項可能要 30~60 秒
DEMOS = [
    {"n": 1, "title": "環境自檢", "slow": False,
     "why": "接手第一件事。缺什麼它會直接說，包含「連線角色是不是 superuser」"
            "—— 用 superuser 連線會繞過 RLS，隔離就形同虛設。",
     "cmd": [PY, "-m", "flowmind.cli", "doctor"]},

    {"n": 2, "title": "回歸測試", "slow": False,
     "why": "不依賴資料庫與 LLM，數秒跑完。要先開 Docker 才能跑的測試，"
            "實務上不會有人跑。其中負向測試組證明的是「錯誤引用會被擋下來」。",
     "cmd": [PY, "tests/test_core.py"]},

    {"n": 3, "title": "★ 造假憑證偵測 + 負向對照組驗收", "slow": False,
     "why": "一套永遠回報「全部通過」的檢查不構成任何證據。"
            "必須先證明它抓得到問題，「它說沒問題」才有意義。",
     "cmd": [PY, "-m", "flowmind.cli", "crosscheck",
             "--tenant", "CASE-9999", "--against-answer-key"]},

    {"n": 4, "title": "乾淨資料的交叉驗證（對照組）", "slow": False,
     "why": "與 demo 3 對照：同一套檢查跑在沒有動過手腳的資料上，"
            "應該全部通過。證明它不是無差別亂報。",
     "cmd": [PY, "-m", "flowmind.cli", "crosscheck", "--tenant", "CASE-0001"]},

    {"n": 5, "title": "★ 資料隔離證明（三態結果）", "slow": False,
     "why": "查詢語句完全沒有 WHERE tenant_id —— 過濾由 PostgreSQL "
            "Row-Level Security 強制執行。內控稽核不接受「我們程式碼有加過濾」。",
     "cmd": [PY, "rag_query.py", "--verify-isolation", "CASE-0001", "CASE-9999"]},

    {"n": 6, "title": "稽核軌跡雜湊鏈驗證", "slow": False,
     "why": "每列串接前一列的雜湊，任何事後竄改或刪除都會斷鏈（tamper-evident）。",
     "cmd": [PY, "rag_query.py", "--verify-audit"]},

    {"n": 7, "title": "委任案清單", "slow": False,
     "why": "engagement 是會計師事務所的用語，不是工程師的 project —— "
            "一個客戶可能同時有多個委任案，這才是隔離的最小單位。",
     "cmd": [PY, "-m", "flowmind.cli", "engagements"]},

    {"n": 8, "title": "★ 統計表決定性查詢", "slow": False,
     "why": "入庫摘要寫著「完整數據請查原始檔案」—— 這支程式讓那句話真的被兌現。"
            "數字直接從原始 xlsx 讀出，未經語言模型。",
     "cmd": [PY, "-m", "flowmind.tables", "機械設備製造業"]},

    {"n": 9, "title": "★ 合成資料的真實統計校準", "slow": False,
     "why": "回答「你們的合成資料憑什麼說它像真的」。"
            "分布權重取自信保基金真實承保統計，覆蓋率不足的部分刻意列出來給自己難看。",
     "cmd": [PY, "-m", "flowmind.calibration"]},

    {"n": 10, "title": "ROI 三情境敏感度", "slow": False,
     "why": "每案人工核對時數查不到官方數字、本來就不是固定值。"
            "做成可調參數的模型，銀行能帶入自己的經驗值重算。",
     "cmd": [PY, "scripts/roi_model.py", "--all-scenarios"]},

    {"n": 11, "title": "ROI 假設與出處", "slow": False,
     "why": "逐項列出每個數字的推算邏輯、方向性佐證、以及來源的侷限。"
            "明確區分「推算」與「量測」。",
     "cmd": [PY, "scripts/roi_model.py", "--show-assumptions"]},

    {"n": 12, "title": "★ 決定性運算路徑（零 LLM）", "slow": True,
     "why": "「最大買方占營收多少」需要跨 90 張發票加總 —— "
            "RAG 取 top-k chunk 在設計上就答不了，只會拒答或編數字。"
            "命中決定性路由後直接用程式算，信心 1.00。",
     "cmd": [PY, "rag_query.py", "--tenant", "CASE-0001",
             "-q", "本案最大買方占營收多少？逾期狀況如何？"]},

    {"n": 13, "title": "★ 可驗證引用（RAG 路徑）", "slow": True,
     "why": "模型的每個主張都附逐字摘錄，程式回原文做字串比對，"
            "對不上的直接從答案移除。此路徑無任何 LLM 參與。",
     "cmd": [PY, "rag_query.py", "--tenant", "CASE-0001", "--top-k", "6",
             "-q", "信保基金的供應商融資，保證成數最高幾成？可以憑哪些文件申請撥貸？"]},

    {"n": 14, "title": "★ 拒答行為（知識庫外的問題）", "slow": True,
     "why": "問一個知識庫不可能有答案的問題。正確行為是拒答並說明缺什麼，"
            "而不是給一個聽起來很篤定的答案。",
     "cmd": [PY, "rag_query.py", "--tenant", "CASE-0001", "--top-k", "6",
             "-q", "2027 年信保基金的保證成數上限會調整到幾成？"]},
]


def run(demo: dict, timeout: int = 900) -> dict:
    t0 = time.time()
    try:
        p = subprocess.run(demo["cmd"], cwd=ROOT, capture_output=True,
                           timeout=timeout, encoding="utf-8", errors="replace")
        out = (p.stdout or "") + (("\n[stderr]\n" + p.stderr) if p.stderr.strip() else "")
        rc = p.returncode
    except subprocess.TimeoutExpired:
        out, rc = f"[逾時 {timeout}s]", -1
    except Exception as e:                             # noqa: BLE001
        out, rc = f"[執行失敗] {e}", -1
    return {**demo, "output": out.rstrip(), "rc": rc,
            "elapsed": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", type=int)
    ap.add_argument("--skip-slow", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    todo = DEMOS
    if args.only:
        todo = [d for d in todo if d["n"] in args.only]
    if args.skip_slow:
        todo = [d for d in todo if not d["slow"]]

    print(f"▶ 執行 {len(todo)} 項 demo\n")
    results = []
    for d in todo:
        print(f"  [{d['n']:>2}] {d['title']} …", end=" ", flush=True)
        r = run(d)
        results.append(r)
        print(f"{'✅' if r['rc'] == 0 else '❌ rc=' + str(r['rc'])}　{r['elapsed']}s")

    ok = sum(1 for r in results if r["rc"] == 0)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    L = [
        "# FlowMind AI — Demo 實測結果",
        "",
        f"> 執行時間：**{now}**　｜　{ok} / {len(results)} 項成功",
        "> ",
        "> 本檔案由 `python scripts/capture_demos.py` 自動產生，內容為**真實終端機輸出**。",
        "> 簡報上放示意圖任何人都做得出來；放真實輸出，代表這套系統當下真的跑得起來。",
        "> ",
        "> **失敗的項目會如實記錄，不會被跳過** ——",
        "> 一份只記錄成功項目的報告，跟沒有報告一樣。",
        "",
        "## 摘要",
        "",
        "| # | 項目 | 結果 | 耗時 |",
        "|---|---|---|---|",
    ]
    for r in results:
        status = "✅ 成功" if r["rc"] == 0 else f"❌ 失敗 (rc={r['rc']})"
        L.append(f"| {r['n']} | {r['title']} | {status} | {r['elapsed']}s |")
    L.append("")
    L.append("---")
    L.append("")

    for r in results:
        L += [f"## {r['n']}. {r['title']}", "",
              f"**為什麼要看這一項**：{r['why']}", "",
              "```powershell",
              " ".join(("python" if c == PY else c) for c in r["cmd"]),
              "```", "",
              f"<sub>執行結果（rc={r['rc']}，耗時 {r['elapsed']}s）</sub>", "",
              "```", r["output"][:6000], "```", "",
              "---", ""]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")

    print(f"\n{'═'*70}")
    print(f"  {ok}/{len(results)} 項成功　→ {out}")
    if ok < len(results):
        print("  ⚠️  失敗項目已如實記錄在報告中：")
        for r in results:
            if r["rc"] != 0:
                print(f"      [{r['n']}] {r['title']}")
    print("═" * 70)


if __name__ == "__main__":
    main()
