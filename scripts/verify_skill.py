#!/usr/bin/env python3
"""
verify_skill.py — 用我們自己的驗證器，驗我們自己的技能檔
=============================================================================
【為什麼這支腳本值得存在】

整個產品的核心主張是「模型說的每一句話都要能回原文逐字比對」。
那麼一個明顯的問題是：**我們自己寫的技能檔，做得到嗎？**

技能檔（SKILL.md）是要交給 AI 助理當作領域知識載入的。
它如果含有無法查證的數字，那個錯誤會被複製到之後每一次的回答裡 ——
比模型單次幻覺嚴重得多，因為它是**系統性**的。

所以這支腳本把技能檔當成「一份待驗證的答案」，
用與線上問答完全相同的規則檢查它：

    標成 📗 原文 的內容 → 必須能在知識庫文件中找到逐字對應
    標成 📊 實測 的數字 → 必須能在原始統計檔案中找到
    標成 🧠 推論 的內容 → 不要求逐字對應，但**必須有這個標記**
    沒有任何標記的具體數字 → 這才是真正的問題

最後一項是重點。一個「沒有標記的數字」比一個「標錯的數字」危險，
因為前者連被檢查的機會都沒有。

【刻意不使用 LLM】
用 LLM 來判斷「這句話是否忠於原文」，等於用一個會出錯的東西
去驗證另一個會出錯的東西。這裡全部用字串比對。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flowmind import config, db                          # noqa: E402

SKILL = Path("skills/taiwan-supply-chain-finance/SKILL.md")

# 逐字引用：用「」或《》或直角引號框起來的片段
_QUOTED = re.compile(r"[「『]([^」』]{6,120})[」』]")

# 具體數字：金額、成數、百分比、天數、年數、條號
_CONCRETE = re.compile(
    r"(?:"
    r"\d[\d,]{2,}\s*(?:萬|億|元|萬元|億元)"      # 金額
    r"|[一二三四五六七八九十]成"                   # 成數
    r"|\d+(?:\.\d+)?\s*%"                        # 百分比
    r"|百分之[零一二三四五六七八九十百]+"
    r"|\d+\s*(?:天|日)"                          # 天期
    r"|第\s*\d+\s*條"                            # 條號
    r")")

MARKERS = {"📗": "原文", "📊": "實測", "🧠": "推論", "❓": "未涵蓋"}

# 上下文視窗：數字前後各取多少字去找關鍵詞。
# 200 字約是一個統計表格列或一個段落的長度 —— 夠寬到能容納
# 「台北市 …… 18.36%」這種中間隔了欄位的排版，
# 又窄到不會把整份文件都算成「上下文」。
CONTEXT_WINDOW = 200


def _numeric_forms(token: str) -> list[str]:
    """
    一個數字在不同檔案裡可能有的等價寫法。
    47.20 / 47.2 / 47 是同一個值的三種排版，字串比對會全部當成不同。
    """
    forms = {token}
    try:
        v = float(token)
    except ValueError:
        return list(forms)
    forms.add(repr(v).rstrip("0").rstrip(".") if "." in token else token)
    forms.add(f"{v:g}")
    if v == int(v):
        forms.add(str(int(v)))
    return sorted(forms, key=len, reverse=True)


def _propagate_markers(lines: list[str]) -> list[list[str]]:
    """
    把出處標記沿續到緊接其後的表格／清單區塊。

    回傳與 lines 等長的清單，每個元素是該行**有效**的標記。
    空行、標題、水平線會中止沿續 —— 沿續得太遠會讓標記失去意義，
    整份文件都算「有標記」等於沒有標記。
    """
    out: list[list[str]] = []
    carry: list[str] = []
    for raw in lines:
        line = raw.strip()
        own = [m for m in MARKERS if m in raw]
        if own:
            carry = own
            out.append(own)
            continue
        if not line:
            # 空行**不**中止沿續：markdown 的正常寫法就是
            # 「📊 標記」→ 空行 →「表格」。一開始把空行當中止，
            # 結果整張統計表都被判成未標記。
            out.append([])
            continue
        if line.startswith(("#", "---", "===", "```")):
            carry = []                       # 標題／水平線／程式區塊才中止
            out.append([])
            continue
        if line.startswith(("|", "-", "*", ">")) and carry:
            out.append(list(carry))          # 表格列／清單項，沿用上方標記
            continue
        carry = []
        out.append([])
    return out


def _positions(haystack: str, needle: str, limit: int = 50):
    """回傳 needle 在 haystack 中出現的位置（最多 limit 個）。"""
    start, n = 0, 0
    while n < limit:
        i = haystack.find(needle, start)
        if i < 0:
            return
        yield i
        start, n = i + 1, n + 1


def _norm(s: str) -> str:
    """全形半形、空白統一。技能檔與原始文件的排版習慣經常不同。"""
    s = unicodedata.normalize("NFKC", s or "")
    return re.sub(r"\s+", "", s)


def load_corpus() -> list[tuple[str, str]]:
    """把知識庫全文抓出來做逐字比對。回傳 [(來源, 正規化後全文)]。"""
    out: list[tuple[str, str]] = []
    with db.tenant_session("SHARED") as conn:
        with conn.cursor() as cur:
            # 表名是 documents（每列一個 chunk），不是 chunks
            cur.execute("""
                SELECT source, string_agg(content, ' ' ORDER BY chunk_index)
                FROM documents WHERE tenant_id = 'SHARED'
                GROUP BY source
            """)
            for src, body in cur.fetchall():
                out.append((src, _norm(body or "")))
    if not out:
        raise RuntimeError(
            "知識庫是空的 —— 無法驗證。先跑 ingest.py 建立 SHARED 語料。")
    return out


def load_stat_values() -> list[tuple[str, str]]:
    """
    原始統計檔案的內容，供 📊 實測數字比對。回傳 [(檔名, 正規化內容)]。

    **必須同時讀 CSV 與 XLSX。**
    一開始只讀 *.csv，於是信保基金的月報（全部是 .xlsx）完全沒被納入，
    技能檔裡那些其實正確的統計數字被報成「驗不到」。
    一個只看得到一半資料的驗證器，產生的假警報會讓人開始不信任它，
    然後真正的錯誤就會混在假警報裡被一起忽略 —— 那比沒有驗證器更糟。
    """
    out: list[tuple[str, str]] = []
    base = config.DATA_DIR / "raw" / "SHARED"
    for p in sorted(base.glob("*.csv")):
        try:
            out.append((p.name, _norm(p.read_text(encoding="utf-8-sig"))))
        except Exception:                                # noqa: BLE001
            continue
    for p in sorted(base.glob("*.xlsx")):
        try:
            import openpyxl                              # noqa: PLC0415
            wb = openpyxl.load_workbook(p, data_only=True, read_only=True)
            cells = []
            for ws in wb.worksheets:
                for row in ws.iter_rows(values_only=True):
                    cells += [str(c) for c in row if c not in (None, "")]
            out.append((p.name, _norm(" ".join(cells))))
        except Exception:                                # noqa: BLE001
            continue
    if not out:
        raise RuntimeError("找不到任何原始統計檔案（CSV/XLSX）—— 無法驗證 📊 數字。")
    return out


def verify(skill_path: Path) -> dict:
    text = skill_path.read_text(encoding="utf-8")
    corpus = load_corpus()
    stats = load_stat_values()
    lines = text.splitlines()

    quoted_results, unmarked = [], []

    # 標記會「沿續」到緊接其後的表格或清單區塊。
    # 實際寫法是把 📊 標在表格上方一行，數字放在表格列裡 ——
    # 一開始逐行判斷，結果那些列既被誤判成「未標記」，
    # 又沒被納入 📊 的驗證，等於兩邊都漏掉。
    # 這是驗證器本身的缺陷，不是技能檔的問題，先修工具再看結果。
    effective = _propagate_markers(lines)

    for lineno, line in enumerate(lines, 1):
        marks = effective[lineno - 1]

        # ── ① 逐字引用必須找得到 ──────────────────────────────────────
        if "📗" in marks:
            for m in _QUOTED.finditer(line):
                frag = _norm(m.group(1))
                if len(frag) < 6:
                    continue
                hit = next((src for src, body in corpus if frag in body), None)
                quoted_results.append({
                    "line": lineno, "fragment": m.group(1)[:80],
                    "found_in": hit, "verified": bool(hit)})

        # ── ② 沒有任何標記、卻含具體數字的行 ──────────────────────────
        # 這是本腳本最重要的檢查：沒標記的數字連被檢查的機會都沒有。
        if not marks:
            nums = _CONCRETE.findall(line)
            # 表格分隔線、標題、目錄不算
            if nums and not line.strip().startswith(("|---", "#", ">", "```")):
                unmarked.append({"line": lineno, "text": line.strip()[:110],
                                 "numbers": nums[:4]})

    # ── ③ 📊 實測數字要在原始統計或文件裡找得到，**且要在對的上下文裡** ──
    #
    # 一開始只檢查「這個數字有沒有出現在語料的某處」，那個檢查太寬鬆：
    # 83 份文件裡一個四位數字幾乎必然會撞到，於是全部都「通過」。
    # 一個永遠會通過的檢查，等於沒有檢查。
    #
    # 改成上下文驗證：數字必須與該行的關鍵詞出現在同一個視窗內。
    # 這仍然不是完美的（同一段落裡出現兩個統計仍可能誤判），
    # 但它把「巧合撞到」這個最主要的假陽性來源擋掉了。
    measured = []
    for lineno, line in enumerate(lines, 1):
        if "📊" not in effective[lineno - 1]:
            continue
        keys = [k for k in re.findall(r"[一-鿿]{2,6}", line)
                if k not in ("實測", "原文", "推論", "未涵蓋")][:6]
        for m in re.finditer(r"\d[\d,]{3,}(?:\.\d+)?|\d+\.\d{2}", line):
            raw = m.group(0).replace(",", "")
            # 數值等價的所有寫法都要試。
            # 技能檔為了排版整齊寫 47.20，原始檔存的是 47.2 ——
            # 純字串比對會把這個判成「查無此數字」。
            # 一個把格式差異報成資料錯誤的驗證器，會製造大量假警報，
            # 而假警報最終會讓人連真警報一起忽略。
            forms = _numeric_forms(raw)
            ok, where = False, None
            for src, body in corpus + stats:
                b = body.replace(",", "")
                for form in forms:
                    for pos in _positions(b, form):
                        window = b[max(0, pos - CONTEXT_WINDOW):
                                   pos + CONTEXT_WINDOW]
                        if any(k in window for k in keys):
                            ok, where = True, src
                            break
                    if ok:
                        break
                if ok:
                    break
            measured.append({"line": lineno, "value": m.group(0),
                             "context_keys": keys[:3],
                             "found_in": where, "verified": ok})

    n_q = len(quoted_results)
    n_qok = sum(1 for r in quoted_results if r["verified"])
    n_m = len(measured)
    n_mok = sum(1 for r in measured if r["verified"])

    return {
        "skill_file": str(skill_path),
        "corpus_documents": len(corpus),
        "quoted_total": n_q,
        "quoted_verified": n_qok,
        "quoted_rate": round(n_qok / n_q, 4) if n_q else None,
        "measured_total": n_m,
        "measured_verified": n_mok,
        "measured_rate": round(n_mok / n_m, 4) if n_m else None,
        "unmarked_concrete_lines": len(unmarked),
        "failures_quoted": [r for r in quoted_results if not r["verified"]],
        "failures_measured": [r for r in measured if not r["verified"]],
        "unmarked": unmarked,
    }


def render(r: dict) -> str:
    L = ["═" * 78,
         "  技能檔自我驗證　（用產品自己的規則，驗產品自己的知識檔）",
         "═" * 78,
         f"  檔案　{r['skill_file']}",
         f"  比對語料　{r['corpus_documents']} 份文件全文", ""]

    def pct(v):
        return "—" if v is None else f"{v:.1%}"

    L += [f"  📗 逐字引用　{r['quoted_verified']}/{r['quoted_total']}"
          f"　通過率 {pct(r['quoted_rate'])}",
          f"  📊 實測數字　{r['measured_verified']}/{r['measured_total']}"
          f"　通過率 {pct(r['measured_rate'])}",
          f"  ⚠️ 未標記卻含具體數字的行　{r['unmarked_concrete_lines']} 行", ""]

    if r["failures_quoted"]:
        L.append("  ── 逐字引用對不上原文 ──")
        for f in r["failures_quoted"][:10]:
            L.append(f"    L{f['line']}　{f['fragment']}")
    if r["failures_measured"]:
        L.append("  ── 實測數字在原始統計中找不到 ──")
        for f in r["failures_measured"][:10]:
            L.append(f"    L{f['line']}　{f['value']}")
    if r["unmarked"]:
        L.append("  ── 未標記出處卻含具體數字（最需要處理的一類）──")
        L.append("     沒標記的數字連被檢查的機會都沒有，")
        L.append("     它的錯誤會被複製到之後每一次回答裡。")
        for u in r["unmarked"][:12]:
            L.append(f"    L{u['line']}　{u['numbers']}　{u['text']}")

    L += ["", "═" * 78,
          "  全部檢查為字串比對，未使用 LLM ——",
          "  用一個會出錯的東西去驗證另一個會出錯的東西沒有意義。",
          "═" * 78]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill", default=str(SKILL))
    ap.add_argument("--out", default="docs/SKILL_VERIFICATION.json")
    args = ap.parse_args()

    r = verify(Path(args.skill))
    print(render(r))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n📄 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
