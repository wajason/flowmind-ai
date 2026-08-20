#!/usr/bin/env python3
"""
run_verifin.py — VeriFin 評測執行器
=============================================================================
把本地 Ollama 模型放到 SROIE / FUNSD / CORD 三份外部 benchmark 上跑，
用 flowmind/verifin.py 的四項指標評分。

【引用區間怎麼來的：不要求模型回報字元位置】
要一個語言模型準確數出「第 1,847 個字元」是不切實際的，它會亂給數字，
那樣測到的是它會不會數數，不是它有沒有讀文件。
改成要求它回傳一段**逐字摘錄**，再由評測程式自己去原文裡找位置。
找不到 → 引用視為未驗證。這一步完全在程式端完成，模型沒有介入空間，
不可 gameable 的性質完全保留，同時把不合理的要求拿掉了。

用法：
  python scripts/run_verifin.py --suite sroie --limit 50
  python scripts/run_verifin.py --suite all --counterfactual
  python scripts/run_verifin.py --suite sroie --model qwen3.5:9b   # 換模型比較
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flowmind import config, llm, verifin                       # noqa: E402
from flowmind.verifin import FieldPrediction                    # noqa: E402

BENCH = config.BENCH_DIR

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "fields": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "field": {"type": "string"},
                    "value": {"type": ["string", "null"]},
                    "confidence": {"type": "number"},
                    "evidence_quote": {"type": ["string", "null"]},
                },
                "required": ["field", "value", "confidence", "evidence_quote"],
            },
        }
    },
    "required": ["fields"],
}

SYSTEM = """你是文件抽取引擎。你的輸出會被程式直接驗證，不會有人幫你潤飾。

規則（違反其中任何一條都會被自動扣分）：
1. 只從提供的文件文字中抽取。文件沒有明確寫出的欄位，value 一律填 null。
2. 嚴禁推測、補完、或用常識填空。看起來「應該是」不等於「文件寫了」。
3. 每個 value 都必須附上 evidence_quote：從文件中「一字不差」複製的片段，
   且該片段必須包含你填的 value。你複製的字串會被程式拿去原文比對，改一個字都會失敗。
4. confidence 填 0 到 1 之間的實數，代表你對這個值的真實把握程度。
   注意計分方式：答對 +1、留白 0、答錯 -2。
   把握度低於 67% 時，留白的期望得分高於猜測。誠實回報對你有利。
5. value 填 null 時，evidence_quote 也填 null，confidence 填 0。"""


def build_prompt(text: str, fields: list[str]) -> str:
    field_list = "\n".join(f"  - {f}" for f in fields)
    return (f"【文件文字】\n{text[:6000]}\n\n"
            f"【要抽取的欄位】\n{field_list}\n\n"
            f"請針對上列每一個欄位各輸出一筆結果（即使是 null 也要輸出）。")


def run_doc(row: dict, model: str) -> tuple[list[FieldPrediction], dict]:
    # timeout 刻意收緊到 150s（預設 600s，retries=1 時最壞情況要等 1200s）。
    # SROIE/FUNSD/CORD 都是短篇收據/表單，實測每份 10~20s；一份卡到 150s
    # 都拿不到回應，代表這次請求本身有問題，繼續等只是在浪費一整晚的時間，
    # 不會讓它突然成功。
    obj, diag = llm.extract_json(
        build_prompt(row["text"], row["ask_fields"]),
        schema=EXTRACT_SCHEMA, system=SYSTEM, model=model, timeout=150,
    )
    preds: list[FieldPrediction] = []
    got = {}
    if isinstance(obj, dict):
        for item in obj.get("fields", []) or []:
            if isinstance(item, dict) and item.get("field"):
                got[str(item["field"])] = item

    for fname in row["ask_fields"]:
        item = got.get(fname, {})
        value = item.get("value")
        quote = item.get("evidence_quote")
        try:
            conf = float(item.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0

        # 引用區間由程式在原文中定位，不採用模型自報的位置
        span = None
        if quote and not verifin.is_abstain(value):
            idx = row["text"].find(str(quote))
            if idx < 0:
                # 大小寫/空白差異的第二次嘗試。這是唯一放寬的地方，
                # 而且是雙方都做同樣的正規化，不是模糊比對。
                lowered = row["text"].lower()
                idx = lowered.find(str(quote).lower().strip())
            if idx >= 0:
                span = (idx, idx + len(str(quote)))

        preds.append(FieldPrediction(field=fname, value=value,
                                     confidence=max(0.0, min(1.0, conf)),
                                     evidence_span=span))
    return preds, diag


def load_rows(suite: str, limit: int | None) -> list[dict]:
    path = BENCH / f"{suite}.jsonl"
    if not path.exists():
        print(f"❌ 找不到 {path}。請先執行 python scripts/fetch_benchmarks.py")
        sys.exit(1)
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[:limit] if limit else rows


def run_doc_safe(row: dict, model: str) -> tuple[list[FieldPrediction], dict]:
    """
    `run_doc()` 包一層容錯。

    【為什麼要這一層】一次全量重跑要對 500+ 份文件各打一次 Ollama，
    跑好幾個小時。**曾經真實發生過**：其中一份文件的請求卡住不動
    （不是報錯，是網路層級掛住，httpx 逾時前完全沒有任何輸出），
    在沒有這層容錯的版本裡，這一份文件會讓前面已經跑完的幾百份
    全部白費——一個晚上的算力，因為一份文件而歸零。

    逾時或任何例外都當成「這份文件抽取失敗」處理：全部欄位計為留白
    （而不是憑空造一個答案），並在 diag 裡老實記下失敗原因。
    這不是把失敗藏起來——`strict_json_failure_rate` 本來就是要如實
    反映「pipeline 真實中斷率」的指標，一次逾時就該算進那個分母，
    不能因為它換了一種失敗方式就不計分。
    """
    try:
        return run_doc(row, model)
    except Exception as e:                                     # noqa: BLE001
        print(f"   ⚠️ {row.get('doc_id', '?')} 抽取失敗（{type(e).__name__}: {e}），"
              f"計為留白後繼續", flush=True)
        preds = [FieldPrediction(field=f, value=None, confidence=0.0, evidence_span=None)
                 for f in row["ask_fields"]]
        return preds, {"strict": False, "error": f"{type(e).__name__}: {e}"}


def evaluate(suite: str, model: str, limit: int | None,
             do_counterfactual: bool, seed: int) -> dict:
    rows = load_rows(suite, limit)
    rng = random.Random(seed)

    print(f"\n▶ {suite.upper()}：{len(rows)} 份文件　模型 {model}")
    results, strict_fail, t0 = [], 0, time.time()
    for i, row in enumerate(rows, 1):
        preds, diag = run_doc_safe(row, model)
        if not diag.get("strict"):
            strict_fail += 1
        results.append(verifin.score_document(
            row["doc_id"], row["dataset"], row["text"], preds, row["gold"]))
        if i % 10 == 0 or i == len(rows):
            print(f"   {i}/{len(rows)}　已用時 {time.time()-t0:.0f}s", flush=True)

    cf_results, changed_map = None, None
    if do_counterfactual:
        print(f"\n▶ {suite.upper()} 反事實擾動（種子 {seed}，每次執行的擾動值都不同）")
        cf_results, changed_map = [], {}
        for i, row in enumerate(rows, 1):
            cf_text, cf_gold, changed = verifin.make_counterfactual(
                row["text"], row["gold"], rng)
            if not changed:
                continue
            changed_map[row["doc_id"]] = changed
            cf_row = {**row, "text": cf_text, "gold": cf_gold}
            preds, _ = run_doc_safe(cf_row, model)
            cf_results.append(verifin.score_document(
                row["doc_id"] + "::cf", row["dataset"], cf_text, preds, cf_gold))
            if i % 10 == 0:
                print(f"   {i}/{len(rows)}", flush=True)

    rep = verifin.build_report(suite.upper(), model, results, cf_results, changed_map)
    rep["elapsed_s"] = round(time.time() - t0, 1)
    # 這個數字要如實報告：需要靠正則搶救才拿得到 JSON 的比例，
    # 直接對應真實客戶現場的 pipeline 中斷率。
    rep["strict_json_failure_rate"] = round(strict_fail / max(1, len(rows)), 4)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="sroie",
                    choices=["sroie", "funsd", "cord", "all"])
    ap.add_argument("--model", default=None, help=f"預設 {config.EXTRACT_MODEL}")
    ap.add_argument("--limit", type=int, default=50,
                    help="每個 suite 取幾份文件。正式數據請用 --limit 0（全量）")
    ap.add_argument("--counterfactual", action="store_true",
                    help="加跑反事實擾動測試（會讓執行時間加倍）")
    ap.add_argument("--seed", type=int, default=None,
                    help="擾動亂數種子。預設用當下時間 —— "
                         "刻意如此，讓每次評測的擾動值都是新的，無法事先記憶")
    ap.add_argument("--out", default="out/verifin_report.json")
    args = ap.parse_args()

    model = args.model or config.EXTRACT_MODEL
    limit = None if args.limit == 0 else args.limit
    seed = args.seed if args.seed is not None else int(time.time())

    if not llm.ollama_available():
        print("❌ 連不上 Ollama。請先確認 ollama 服務已啟動（ollama serve）。")
        sys.exit(1)

    suites = ["sroie", "funsd", "cord"] if args.suite == "all" else [args.suite]
    reports = []
    for s in suites:
        rep = evaluate(s, model, limit, args.counterfactual, seed)
        print("\n" + verifin.render_report(rep))
        print(f"   嚴格 JSON 失敗率：{rep['strict_json_failure_rate']:.1%}"
              f"　總耗時 {rep['elapsed_s']}s")
        reports.append(rep)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seed": seed, "reports": reports},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n📄 完整報告 → {out}")


if __name__ == "__main__":
    main()
