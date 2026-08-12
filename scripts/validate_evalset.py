#!/usr/bin/env python3
"""
validate_evalset.py — 答案卷本身也要被驗證
=============================================================================
【為什麼需要這支腳本】

擴充評測題庫時最容易犯、也最難發現的錯，是**寫出一份錯的答案卷**：
`expect_source` 指到不存在的檔案、`expect_contains` 寫了一段原文根本
沒有的文字。

那種錯誤的後果特別惡劣：系統答對了卻被判錯，於是有人去「修」一個
根本沒壞的東西；或是反過來，把一個錯的行為當成正確基準鎖死。

**一份沒有被驗證過的答案卷，會讓所有用它做出來的結論失效。**

所以每一題都檢查三件事：
    ① expect_source 指的檔案確實在知識庫裡
    ② 每個 expect_contains 片段確實在**該檔案**中逐字存在
    ③ must_abstain 的題目不得同時給 expect_contains（語意上矛盾）

第 ② 點刻意要求「在該檔案中」而不是「在語料的某處」——
後者太寬鬆，83 份文件裡一個常見詞幾乎必然找得到。

Usage:
    python scripts/validate_evalset.py
    python scripts/validate_evalset.py --file data/evalset/zh_finance_qa.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flowmind import db                                    # noqa: E402

DEFAULT = ROOT / "data" / "evalset" / "zh_finance_qa.jsonl"


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s or ""))


def load_corpus() -> dict[str, str]:
    """
    每份文件的可比對全文。

    **統計表要額外從原始檔案讀。**
    XLSX/CSV 統計表在入庫時是刻意「摘要化」的（不逐列入庫，
    精確數值改由 tables.py 直接查原始檔），
    所以 documents 裡沒有那些數字。
    只查 documents 會把正確的答案卷判成錯的 —— 第一版就是這樣，
    12 筆假警報全部來自統計表。
    """
    out: dict[str, str] = {}
    with db.tenant_session("SHARED") as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT source, string_agg(content, ' ' ORDER BY chunk_index) "
                        "FROM documents WHERE tenant_id = 'SHARED' GROUP BY source")
            for src, body in cur.fetchall():
                out[src] = _norm(body or "")
    if not out:
        raise SystemExit("知識庫是空的 —— 無法驗證答案卷。先跑 data_update_finance.py。")

    raw = ROOT / "data" / "raw" / "SHARED"
    for p in list(raw.glob("*.xlsx")) + list(raw.glob("*.csv")):
        try:
            if p.suffix.lower() == ".csv":
                txt = p.read_text(encoding="utf-8-sig")
            else:
                import openpyxl                             # noqa: PLC0415
                wb = openpyxl.load_workbook(p, data_only=True, read_only=True)
                cells = []
                for ws in wb.worksheets:
                    for row in ws.iter_rows(values_only=True):
                        cells += [str(c) for c in row if c not in (None, "")]
                txt = " ".join(cells)
        except Exception:                                  # noqa: BLE001
            continue
        out[p.name] = out.get(p.name, "") + " " + _norm(txt)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(DEFAULT))
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            Path(args.file).read_text(encoding="utf-8").splitlines() if l.strip()]
    corpus = load_corpus()

    print("═" * 76)
    print("  答案卷驗證　（一份沒被驗證過的答案卷，會讓所有結論失效）")
    print("═" * 76)
    print(f"  題數 {len(rows)}　知識庫 {len(corpus)} 份文件\n")

    bad_src, bad_frag, contradictory, dup = [], [], [], []
    seen_q: Counter = Counter()

    for r in rows:
        qid = r.get("id", "?")
        seen_q[_norm(r.get("q", ""))] += 1

        if r.get("must_abstain") and r.get("expect_contains"):
            contradictory.append((qid, "must_abstain 卻給了 expect_contains"))

        src = r.get("expect_source")
        if not src:
            continue
        if src not in corpus:
            bad_src.append((qid, src))
            continue
        for frag in (r.get("expect_contains") or []):
            # 數值的等價寫法都要試：答案卷為了好讀寫「4,414,900」，
            # 原始統計檔存的是「4414900」。純字串比對會把正確的答案
            # 判成錯的 —— 一個會製造假警報的驗證器，最終會讓人
            # 連真警報一起忽略。
            body = corpus[src]
            forms = {_norm(frag), _norm(frag).replace(",", "")}
            if not any(f and f in body.replace(",", "") or f in body
                       for f in forms):
                bad_frag.append((qid, src, frag))

    dup = [(q, n) for q, n in seen_q.items() if n > 1]

    for label, items, fmt in [
        ("expect_source 指到知識庫沒有的檔案", bad_src,
         lambda x: f"{x[0]}　{x[1]}"),
        ("expect_contains 在該檔案中查無此逐字片段", bad_frag,
         lambda x: f"{x[0]}　{x[2]!r}　（宣稱出自 {x[1]}）"),
        ("must_abstain 與 expect_contains 同時存在（語意矛盾）", contradictory,
         lambda x: f"{x[0]}　{x[1]}"),
        ("重複的題目", dup, lambda x: f"{x[0][:40]}…　出現 {x[1]} 次"),
    ]:
        if items:
            print(f"  ❌ {label}　{len(items)} 筆")
            for x in items[:12]:
                print(f"       {fmt(x)}")
            print()

    n_bad = len(bad_src) + len(bad_frag) + len(contradictory) + len(dup)
    tiers = Counter(r.get("tier") for r in rows)
    srcs = Counter(r.get("expect_source") for r in rows if r.get("expect_source"))
    print(f"  分層　{dict(tiers)}")
    print(f"  必須拒答　{sum(1 for r in rows if r.get('must_abstain'))} 題")
    print(f"  覆蓋文件　{len(srcs)} 份")
    print()
    print("  ✅ 答案卷全部通過驗證" if n_bad == 0
          else f"  ❌ 共 {n_bad} 處問題，必須修正後才能用來評測")
    print("═" * 76)
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
