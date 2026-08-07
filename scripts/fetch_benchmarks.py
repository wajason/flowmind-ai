#!/usr/bin/env python3
"""
fetch_benchmarks.py — 下載並轉換文件抽取公開 benchmark
=============================================================================
【為什麼一定要用外部 benchmark，而不是自己造題自己考】

我們的合成資料產生器可以產生無限多張發票，答案也完全已知。
但「自己出題、自己作答、自己閱卷」的分數，對評審與銀行都沒有說服力 ——
因為出題的人可以（哪怕是無意識地）把題目調整成自己的系統剛好會做的樣子。

SROIE / FUNSD / CORD 的標註是國際研究社群在我們之前就做好的。
我們無法修改標準答案，也無法挑掉做不出來的題目（腳本會固定用整個 test split）。
這是這三個資料集在本專案的唯一價值來源：**標準答案的所有權不在我們手上。**

【三個資料集的適用性評估：不是全部照單全收】

  SROIE (ICDAR 2019)  ★ 主要指標
      1,000 張掃描收據，含 OCR 後的 words 與四個欄位的標準答案。
      適合的關鍵原因：它提供 OCR 後的文字，讓我們可以「單獨」評測
      「非結構化文字 → 結構化欄位」這一步，不被 OCR 引擎的好壞混淆。
      這一步正是 FlowMind 的核心能力。
      已知落差：英文/馬來西亞零售收據，不是中文 B2B 發票。誠實揭露，不宣稱等同。

  FUNSD              ★ 次要指標（穩健性）
      199 份雜訊很大的掃描表單，標註 question-answer 配對關係。
      價值不在欄位本身，而在「表單裡哪一格是標籤、哪一格是值」的結構理解，
      這對應到真實的合約與銀行對帳單。樣本少、雜訊大，正好測穩健性。

  CORD               ★ 反向指標（拒答能力）
      1,000 張印尼零售收據，30 種細項標註（菜單品項、單價…）。
      跟 B2B 供應鏈金融的語意距離最遠 —— 所以我們刻意「反過來用」：
      對 CORD 收據詢問 B2B 專屬欄位（統一編號、付款帳期），
      正確行為是全部留白。這一項不測抽得準不準，測的是「不會硬掰」。
      一個在 CORD 上硬掰出統編的系統，在真實客戶文件上一定也會硬掰。

用法：
    python scripts/fetch_benchmarks.py                # 三個都抓
    python scripts/fetch_benchmarks.py --only sroie
    python scripts/fetch_benchmarks.py --limit 100    # 先小量試跑
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flowmind import config                                    # noqa: E402

OUT = config.BENCH_DIR
OUT.mkdir(parents=True, exist_ok=True)


def _words_to_text(words) -> str:
    """把 OCR token 串回文字。保留原始順序，不做重排 —— 重排等於偷偷幫模型整理版面。"""
    if isinstance(words, list):
        return " ".join(str(w) for w in words if str(w).strip())
    return str(words or "")


# ── SROIE ────────────────────────────────────────────────────────────────
def build_sroie(limit: int | None) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("jsdnrs/ICDAR2019-SROIE", split="test")
    rows = []
    for i, ex in enumerate(ds):
        if limit and i >= limit:
            break
        ent = ex.get("entities") or {}
        if isinstance(ent, str):
            try:
                ent = json.loads(ent)
            except Exception:                          # noqa: BLE001
                ent = {}
        text = _words_to_text(ex.get("words"))
        if not text.strip():
            continue
        rows.append({
            "doc_id": f"sroie-{ex.get('key', i)}",
            "dataset": "SROIE",
            "text": text,
            # 標準答案原封不動照抄，不做任何正規化 —— 正規化留給評分器，
            # 而且評分器的正規化規則要寫死在程式碼裡供人檢查。
            "gold": {k: (v if isinstance(v, str) else None) for k, v in ent.items()},
            "ask_fields": ["company", "date", "address", "total"],
        })
    return rows


# ── FUNSD ────────────────────────────────────────────────────────────────
def build_funsd(limit: int | None) -> list[dict]:
    """
    FUNSD 的標註是 BIO 序列（O / B-HEADER / B-QUESTION / B-ANSWER / I-…）。
    我們把它還原成 question→answer 配對：把連續的 QUESTION token 併成標籤、
    緊接其後的 ANSWER token 併成值。這是還原成「表單鍵值對」最直接的作法。
    """
    from datasets import load_dataset
    ds = load_dataset("nielsr/funsd", split="test")
    names = ds.features["ner_tags"].feature.names          # 直接取官方標籤名，不硬編碼
    rows = []
    for i, ex in enumerate(ds):
        if limit and i >= limit:
            break
        words, tags = ex["words"], ex["ner_tags"]
        pairs, cur_q, cur_a, mode = [], [], [], None
        for w, t in zip(words, tags):
            label = names[t]
            kind = label.split("-")[-1] if "-" in label else label
            if kind == "QUESTION":
                if mode == "ANSWER" and cur_q and cur_a:
                    pairs.append((" ".join(cur_q), " ".join(cur_a)))
                    cur_q, cur_a = [], []
                if label.startswith("B-") and mode == "QUESTION" and cur_q and cur_a:
                    pairs.append((" ".join(cur_q), " ".join(cur_a)))
                    cur_q, cur_a = [], []
                cur_q.append(w)
                mode = "QUESTION"
            elif kind == "ANSWER":
                cur_a.append(w)
                mode = "ANSWER"
            else:
                if cur_q and cur_a:
                    pairs.append((" ".join(cur_q), " ".join(cur_a)))
                cur_q, cur_a, mode = [], [], None
        if cur_q and cur_a:
            pairs.append((" ".join(cur_q), " ".join(cur_a)))

        pairs = [(q, a) for q, a in pairs if len(q) > 2 and len(a) > 0][:6]
        if not pairs:
            continue
        rows.append({
            "doc_id": f"funsd-{ex['id']}",
            "dataset": "FUNSD",
            "text": " ".join(words),
            "gold": {q: a for q, a in pairs},
            "ask_fields": [q for q, _ in pairs],
        })
    return rows


# ── CORD（反向用途：拒答測試）────────────────────────────────────────────
B2B_ONLY_FIELDS = ["buyer_tax_id", "seller_tax_id", "payment_terms_days",
                   "due_date", "contract_number", "purchase_order_number"]


def build_cord(limit: int | None) -> list[dict]:
    """
    CORD 的 ground_truth 是巢狀 JSON。我們只取其中的文字內容當作「文件全文」，
    然後刻意詢問 B2B 發票才有的欄位。這些欄位在零售收據上「確定不存在」，
    所以標準答案全部是 null —— 這是一份標準答案為「全部留白」的考卷。

    為什麼這樣測有意義：抽取任務的分數可以靠多猜來刷高，
    但拒答任務不行 —— 只要猜一個就扣分。兩份考卷合起來才擋得住刷分。
    """
    from datasets import load_dataset
    ds = load_dataset("naver-clova-ix/cord-v2", split="test")
    rows = []
    for i, ex in enumerate(ds):
        if limit and i >= limit:
            break
        gt = ex.get("ground_truth")
        if isinstance(gt, str):
            try:
                gt = json.loads(gt)
            except Exception:                          # noqa: BLE001
                continue
        texts: list[str] = []

        def walk(node):
            if isinstance(node, dict):
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)
            elif isinstance(node, str):
                texts.append(node)

        walk(gt)
        text = " ".join(texts)
        if len(text) < 30:
            continue
        rows.append({
            "doc_id": f"cord-{i}",
            "dataset": "CORD",
            "text": text,
            "gold": {f: None for f in B2B_ONLY_FIELDS},   # 標準答案：全部留白
            "ask_fields": B2B_ONLY_FIELDS,
            "note": "反向測試：零售收據上不存在 B2B 欄位，正確行為是全部回 null",
        })
    return rows


BUILDERS = {"sroie": build_sroie, "funsd": build_funsd, "cord": build_cord}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(BUILDERS))
    ap.add_argument("--limit", type=int, default=None,
                    help="每個資料集最多取幾筆。正式評測請不要設，"
                         "取子集會讓分數失去與其他研究比較的意義")
    args = ap.parse_args()

    targets = [args.only] if args.only else list(BUILDERS)
    summary = []
    for name in targets:
        print(f"→ 建構 {name.upper()} …", flush=True)
        try:
            rows = BUILDERS[name](args.limit)
        except Exception as e:                         # noqa: BLE001
            print(f"  ❌ 失敗：{e}")
            summary.append({"dataset": name, "n": 0, "error": str(e)[:200]})
            continue
        path = OUT / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        avg = sum(len(r["text"]) for r in rows) / max(1, len(rows))
        print(f"  ✅ {len(rows)} 筆 → {path.name}（平均長度 {avg:.0f} 字元）")
        summary.append({"dataset": name, "n": len(rows), "file": path.name,
                        "avg_chars": round(avg)})

    (OUT / "_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n輸出目錄：{OUT}")
    print("下一步： python scripts/run_verifin.py --suite all")


if __name__ == "__main__":
    main()
