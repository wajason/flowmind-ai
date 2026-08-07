"""
flowmind.tables — 統計表的決定性查詢層
=============================================================================
【這支檔案補的是一句謊言。】

入庫時，CSV/XLSX 統計表不逐列進向量庫（那是 CSV-in-RAG 的典型反模式：
檢索結果會變成一堆破碎的數字列，反而傷害語意檢索），
改成用 pandas 產生摘要，摘要最後寫著：

    「完整數據請查原始檔案 xxx.csv」

問題是：**在此之前，沒有任何程式真的會去查。**
那句話對使用者是一個無法兌現的承諾，對 LLM 則是一個它做不到的指示。
它會怎麼辦？它會從摘要裡的「範圍 517,673 ~ 48,577,732」硬湊一個數字出來 ——
這正是我們整套系統在防的事。

所以摘要必須是一個**真的能被跟隨的指標（pointer）**，而不是一句話。
這支模組就是那個「跟隨」的動作：

    向量檢索  →  定位到「哪一份統計表在講這件事」（語意，適合 LLM）
    本模組    →  從原始檔案取出精確數字（算術，不能給 LLM）

兩段分工的界線和系統其他地方一致：**能用算的就不要讓語言模型去猜。**
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from . import config

TABLE_EXTS = {".csv", ".xlsx", ".xls"}


@dataclass
class TableHit:
    source: str          # 原始檔名
    row_label: str       # 命中的列（行業別／縣市別／組織型態…）
    columns: dict[str, Any]
    period: Optional[str] = None
    unit: Optional[str] = None

    def render(self) -> str:
        cols = "、".join(f"{k} {v:,}" if isinstance(v, (int, float)) else f"{k} {v}"
                         for k, v in self.columns.items())
        meta = "".join(x for x in [f"（{self.period}）" if self.period else "",
                                   f"　單位：{self.unit}" if self.unit else ""] if x)
        return f"{self.row_label}：{cols}{meta}"


# ══════════════════════════════════════════════════════════════════════════
# 讀表
# ══════════════════════════════════════════════════════════════════════════

def _read(path: Path):
    import pandas as pd
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, header=None, encoding="utf-8-sig")
    return pd.read_excel(path, header=None)


def _find_header_row(df) -> int:
    """
    找出真正的欄位名稱那一列。

    政府統計表前幾列常是機關名稱、表名、期別、單位，欄位名可能在第 3~6 列。
    判斷方式：第一個「該列非空欄位數 ≥ 2、且下一列第 2 欄是數字」的位置。
    找不到就回 0，讓呼叫端當作無表頭處理 —— 不猜、不硬套。
    """
    for i in range(min(10, len(df) - 1)):
        row = df.iloc[i]
        nonempty = sum(1 for v in row if str(v).strip() not in ("nan", "", "None"))
        if nonempty < 2:
            continue
        nxt = df.iloc[i + 1]
        try:
            float(str(nxt[1]).replace(",", ""))
            return i
        except (TypeError, ValueError, IndexError):
            continue
    return 0


def _meta(df) -> tuple[Optional[str], Optional[str]]:
    """從表頭區抓期別與單位。抓不到回 None —— 查無則留白。"""
    period = unit = None
    # 逐格轉字串，不用 DataFrame.astype(str)：
    # 混合 dtype 的 DataFrame 取 .values 後仍可能拿到 float，
    # 直接丟給 re.search 會 TypeError。
    for v in df.head(6).values.flatten():
        s = str(v)
        if s in ("nan", "None"):
            continue
        if period is None and re.search(r"\d{2,4}\s*年", s):
            period = s.strip()
        if unit is None and "單位" in s:
            unit = s.split(":")[-1].split("：")[-1].strip()
    return period, unit


@lru_cache(maxsize=64)
def _load(path_str: str):
    path = Path(path_str)
    df = _read(path)
    hdr = _find_header_row(df)
    period, unit = _meta(df)
    cols = [str(c).strip() for c in df.iloc[hdr]]
    body = df.iloc[hdr + 1:].reset_index(drop=True)
    body.columns = cols + [f"_{i}" for i in range(len(body.columns) - len(cols))]
    return body, cols, period, unit


# ══════════════════════════════════════════════════════════════════════════
# 查詢
# ══════════════════════════════════════════════════════════════════════════

def list_tables(tenant_dir: str = "SHARED") -> list[Path]:
    d = config.RAW_DIR / tenant_dir
    if not d.exists():
        return []
    return sorted(p for p in d.iterdir()
                  if p.suffix.lower() in TABLE_EXTS and not p.name.startswith("_"))


def _to_num(v) -> Any:
    s = str(v).replace(",", "").strip()
    if s in ("nan", "", "None", "-"):
        return None
    try:
        f = float(s)
        return int(f) if f == int(f) else f
    except ValueError:
        return s


def lookup(keyword: str, tenant_dir: str = "SHARED",
           sources: Optional[list[str]] = None, limit: int = 12) -> list[TableHit]:
    """
    在統計表的第一欄（類別名稱）找關鍵字，回傳該列的精確數值。

    只做字串包含比對，不做語意近似 —— 這一層的價值就在於它是決定性的。
    「這個數字哪來的」必須能回答成「某檔案的某一列」，而不是「模型算的」。
    """
    hits: list[TableHit] = []
    failed: list[tuple[str, str]] = []
    for path in list_tables(tenant_dir):
        if sources and path.name not in sources:
            continue
        try:
            body, cols, period, unit = _load(str(path))
        except Exception as e:                         # noqa: BLE001
            # 不可以靜靜跳過。
            # 這裡原本寫 `except: continue`，結果一個 TypeError 讓所有表都讀不進來，
            # 而使用者看到的是「查無資料」—— 程式壞掉偽裝成查詢結果，
            # 這正是本專案一直在防的那種靜默失敗。
            failed.append((path.name, str(e)[:80]))
            continue

        first = cols[0] if cols else body.columns[0]
        for _, row in body.iterrows():
            label = str(row[first]).strip()
            if label in ("nan", "", "None") or keyword not in label:
                continue
            vals = {}
            for c in cols[1:]:
                n = _to_num(row.get(c))
                if n is not None:
                    vals[c] = n
            if vals:
                hits.append(TableHit(source=path.name, row_label=label,
                                     columns=vals, period=period, unit=unit))
            if len(hits) >= limit:
                return hits

    if failed:
        print(f"  ⚠️  {len(failed)} 張統計表讀取失敗，這些表未納入查詢範圍：")
        for name, err in failed[:5]:
            print(f"      {name}：{err}")
    return hits


def describe(source: str, tenant_dir: str = "SHARED") -> Optional[dict]:
    """回傳某張表的結構描述，供使用者知道能查什麼。"""
    path = config.RAW_DIR / tenant_dir / source
    if not path.exists():
        return None
    body, cols, period, unit = _load(str(path))
    first = cols[0] if cols else body.columns[0]
    labels = [str(v).strip() for v in body[first]
              if str(v).strip() not in ("nan", "", "None")]
    return {"source": source, "period": period, "unit": unit,
            "columns": cols, "row_count": len(labels),
            "row_labels_sample": labels[:20]}


@lru_cache(maxsize=4)
def _row_label_index(tenant_dir: str = "SHARED") -> tuple[str, ...]:
    """
    蒐集所有統計表的第一欄類別名稱，作為「問題裡提到哪個類別」的比對字典。

    用實際存在的列標籤當字典，而不是叫模型抽關鍵字 ——
    這樣「查得到」與「查不到」的界線是明確的：
    問題裡如果沒有出現任何一個真實存在的類別名稱，就是查不到，
    不會有一個模型猜出來的近似詞把查詢導到錯的列上。
    """
    labels: set[str] = set()
    for path in list_tables(tenant_dir):
        try:
            body, cols, _, _ = _load(str(path))
        except Exception:                              # noqa: BLE001
            continue
        first = cols[0] if cols else body.columns[0]
        for v in body[first]:
            s = str(v).strip()
            if 2 <= len(s) <= 30 and s not in ("nan", "None", "合計", "總計"):
                labels.add(s)
    # 長的優先：「機械設備製造業」要贏過「製造業」
    return tuple(sorted(labels, key=len, reverse=True))


def match_question(question: str, tenant_dir: str = "SHARED",
                   max_terms: int = 3) -> list[str]:
    """從問題中找出實際存在於統計表的類別名稱。找不到就回空 list。"""
    q = re.sub(r"\s+", "", question)
    found: list[str] = []
    for label in _row_label_index(tenant_dir):
        if label in q and not any(label in f for f in found):
            found.append(label)
        if len(found) >= max_terms:
            break
    return found


def render_hits(hits: list[TableHit], keyword: str) -> str:
    if not hits:
        return (f"統計表中查無「{keyword}」。\n"
                f"（此為決定性查詢的結果 —— 查無就是查無，不會由模型推估一個數字。）")
    by_src: dict[str, list[TableHit]] = {}
    for h in hits:
        by_src.setdefault(h.source, []).append(h)
    L = [f"### 統計表精確數值查詢：「{keyword}」", ""]
    for src, rows in by_src.items():
        L.append(f"**{src}**")
        for h in rows:
            L.append(f"- {h.render()}")
        L.append("")
    L.append("*以上數字由程式直接從原始統計檔案讀出，未經語言模型處理。*")
    return "\n".join(L)


if __name__ == "__main__":
    import sys
    kw = sys.argv[1] if len(sys.argv) > 1 else "機械"
    print(render_hits(lookup(kw), kw))
