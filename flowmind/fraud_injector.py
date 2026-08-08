"""
flowmind.fraud_injector — 造假樣態注入器（評測用）
=============================================================================
【為什麼需要這支程式】

一套永遠回報「全部通過」的檢查不構成任何證據。
要證明檢查有效，必須先證明它抓得到問題 —— 而且要能算出
**precision / recall / F1**，不是「五項全中」這種沒有分母的說法。

真實造假案例不會被公開標註，所以唯一能建立 ground truth 的方法，
就是**自己注入已知的瑕疵**。這支程式定義 22 種造假樣態，
每一種都對應一個真實世界確實發生過的手法，
並記錄「注入了什麼、注入在哪一張」作為答案卷。

【樣態的分級：不是所有造假都一樣難抓】

  Tier 1 單張可判定 —— 只看一張憑證就能發現（統編、算術、日期）
  Tier 2 跨憑證     —— 要看整批才能發現（重複、連號、自我交易）
  Tier 3 跨文件     —— 要比對合約或流水才能發現（帳期美化、虛報收款）
  Tier 4 統計性     —— 要足夠樣本才能發現（班佛、整數偏好、假日開票）
  Tier 5 抓不到     —— 明確標示為**已知抓不到**的樣態，用來誠實揭露能力邊界

Tier 5 特別重要：一份只列出「我們抓得到什麼」的報告是不完整的。
評審與銀行真正想知道的是「什麼抓不到」。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Optional


@dataclass
class Defect:
    """一個被注入的瑕疵。這就是評測的 ground truth。"""
    defect_id: str
    tier: int
    name: str
    expected_check: Optional[str]      # 應該由哪一項檢查抓到；None = 已知抓不到
    target: str                        # 被動手腳的憑證編號
    description: str


@dataclass
class InjectionResult:
    invoices: list[dict]
    contracts: list[dict]
    ledger: list[dict]
    defects: list[Defect] = field(default_factory=list)

    @property
    def answer_key(self) -> list[dict]:
        return [{"defect_id": d.defect_id, "tier": d.tier, "name": d.name,
                 "expected_check": d.expected_check, "target": d.target,
                 "description": d.description} for d in self.defects]


# ══════════════════════════════════════════════════════════════════════════
# 樣態定義：每一個函式注入一種瑕疵，回傳 Defect 或 None
# ══════════════════════════════════════════════════════════════════════════

def _pick(inv: list[dict], rng: random.Random, used: set[str]) -> Optional[dict]:
    """挑一張還沒被動過手腳的發票。避免同一張被注入多種瑕疵而互相干擾。"""
    cands = [i for i in inv if str(i.get("invoice_number")) not in used]
    if not cands:
        return None
    x = rng.choice(cands)
    used.add(str(x["invoice_number"]))
    return x


# ── Tier 1：單張可判定 ────────────────────────────────────────────────

def d_invalid_ban(inv, con, led, rng, used) -> Optional[Defect]:
    t = _pick(inv, rng, used)
    if not t:
        return None
    from .textnorm import validate_tax_id
    old = t["buyer_ban"]
    # 不能只把末碼 +1 就假設檢核碼一定會壞 ——
    # 第 7 位為 7 時檢核碼有兩組合法解，末碼 +1 有機率仍然通過。
    # 首次評測 D01 命中率 90%，那 10% 就是這樣來的：
    # **注入器沒有真的注入瑕疵**，是評測本身的 bug，不是偵測失敗。
    for delta in range(1, 10):
        cand = old[:7] + str((int(old[7]) + delta) % 10)
        if not validate_tax_id(cand):
            t["buyer_ban"] = cand
            return Defect("D01", 1, "買方統編檢核碼錯誤", "TAXID-01",
                          t["invoice_number"],
                          f"統編 {old} → {cand}（人頭買方）")
    return None


def d_arith_mismatch(inv, con, led, rng, used) -> Optional[Defect]:
    t = _pick(inv, rng, used)
    if not t:
        return None
    t["total_amount"] = t["total_amount"] + rng.choice([1000, 5000, 10000])
    return Defect("D02", 1, "金額加總不符", "ARITH-01",
                  t["invoice_number"], "銷售額＋稅額 ≠ 總額（竄改後忘記改總計）")


def d_wrong_tax_rate(inv, con, led, rng, used) -> Optional[Defect]:
    t = _pick(inv, rng, used)
    if not t:
        return None
    t["tax_amount"] = round(t["sales_amount"] * 0.12)
    t["total_amount"] = t["sales_amount"] + t["tax_amount"]
    return Defect("D03", 1, "營業稅率異常", "ARITH-02",
                  t["invoice_number"], "稅額為銷售額 12%，非法定 5%")


def d_future_date(inv, con, led, rng, used) -> Optional[Defect]:
    t = _pick(inv, rng, used)
    if not t:
        return None
    d = date.today() + timedelta(days=rng.randint(10, 60))
    t["invoice_date"] = d.isoformat()
    t["due_date"] = (d + timedelta(days=int(t["payment_terms_days"]))).isoformat()
    return Defect("D04", 1, "發票日期在未來", "DATE-01",
                  t["invoice_number"], "開立日晚於基準日（預開發票衝營收）")


def d_reversed_due(inv, con, led, rng, used) -> Optional[Defect]:
    t = _pick(inv, rng, used)
    if not t:
        return None
    idate = date.fromisoformat(t["invoice_date"])
    t["due_date"] = (idate - timedelta(days=rng.randint(5, 30))).isoformat()
    return Defect("D05", 1, "到期日早於開立日", "DATE-02",
                  t["invoice_number"], "日期邏輯矛盾")


def d_negative_amount(inv, con, led, rng, used) -> Optional[Defect]:
    t = _pick(inv, rng, used)
    if not t:
        return None
    t["total_amount"] = 0
    return Defect("D06", 1, "發票金額為零", "AMT-01",
                  t["invoice_number"], "金額為零（作廢票混入送件）")


def d_negative_tax(inv, con, led, rng, used) -> Optional[Defect]:
    t = _pick(inv, rng, used)
    if not t:
        return None
    t["tax_amount"] = -abs(t["tax_amount"])
    t["total_amount"] = t["sales_amount"] + t["tax_amount"]
    return Defect("D07", 1, "稅額為負", "AMT-02",
                  t["invoice_number"], "負稅額（折讓單誤當發票）")


def d_absurd_term(inv, con, led, rng, used) -> Optional[Defect]:
    t = _pick(inv, rng, used)
    if not t:
        return None
    idate = date.fromisoformat(t["invoice_date"])
    t["payment_terms_days"] = 500
    t["due_date"] = (idate + timedelta(days=500)).isoformat()
    return Defect("D08", 1, "帳期超過一年", "DATE-03",
                  t["invoice_number"], "帳期 500 天，B2B 極罕見")


# ── Tier 2：跨憑證 ────────────────────────────────────────────────────

def d_self_dealing(inv, con, led, rng, used) -> Optional[Defect]:
    t = _pick(inv, rng, used)
    if not t:
        return None
    t["buyer_ban"] = t["seller_ban"]
    t["buyer_name"] = t["seller_name"]
    return Defect("D09", 2, "自我交易", "FRAUD-01",
                  t["invoice_number"], "買賣方統編相同（關係企業虛增營收）")


def d_duplicate_number(inv, con, led, rng, used) -> Optional[Defect]:
    t = _pick(inv, rng, used)
    if not t:
        return None
    dup = dict(t)
    dup["total_amount"] = t["total_amount"] + 1
    inv.append(dup)
    return Defect("D10", 2, "發票號碼重複", "DUP-01",
                  t["invoice_number"], "同一號碼出現兩次且金額不同")


def d_duplicate_billing(inv, con, led, rng, used) -> Optional[Defect]:
    t = _pick(inv, rng, used)
    if not t:
        return None
    dup = dict(t)
    dup["invoice_number"] = f"AB{rng.randint(10_000_000, 99_999_999)}"
    inv.append(dup)
    return Defect("D11", 2, "重複請款", "DUP-02",
                  dup["invoice_number"],
                  f"與 {t['invoice_number']} 同買方/同金額/同日期（向兩家銀行分別融資）")


def d_sequential_numbers(inv, con, led, rng, used) -> Optional[Defect]:
    """把同一買方的發票號碼改成連號 —— 一次補開的痕跡。"""
    from collections import defaultdict
    by_buyer = defaultdict(list)
    for i in inv:
        if str(i.get("invoice_number")) not in used:
            by_buyer[i["buyer_ban"]].append(i)
    target = next((v for v in by_buyer.values() if len(v) >= 4), None)
    if not target:
        return None
    base = rng.randint(20_000_000, 90_000_000)
    for k, x in enumerate(target[:4]):
        x["invoice_number"] = f"AB{base + k}"
        used.add(x["invoice_number"])
    return Defect("D12", 2, "發票號碼連號", "SEQ-01",
                  f"AB{base}~AB{base+3}", "同一買方連續 4 張連號（一次補開）")


def d_related_party_ban(inv, con, led, rng, used) -> Optional[Defect]:
    t = _pick(inv, rng, used)
    if not t:
        return None
    t["buyer_ban"] = t["seller_ban"][:4] + t["buyer_ban"][4:]
    return Defect("D13", 2, "關係企業徵兆", "RELATED-01",
                  t["invoice_number"], "買賣方統編前四碼相同（弱訊號）")


# ── Tier 3：跨文件 ────────────────────────────────────────────────────

def d_term_mismatch(inv, con, led, rng, used) -> Optional[Defect]:
    """帳期與合約不符，但發票內部欄位完全自洽 —— 只有跨文件比對抓得到。"""
    if not con:
        return None
    c = rng.choice(con)
    cands = [i for i in inv if i.get("buyer_ban") == c["buyer_ban"]
             and str(i.get("invoice_number")) not in used]
    if not cands:
        return None
    t = rng.choice(cands)
    used.add(str(t["invoice_number"]))
    new_term = 30 if c["payment_terms_days"] != 30 else 90
    t["payment_terms_days"] = new_term
    t["due_date"] = (date.fromisoformat(t["invoice_date"])
                     + timedelta(days=new_term)).isoformat()
    return Defect("D14", 3, "帳期與合約不符", "TERM-02", t["invoice_number"],
                  f"發票 {new_term} 天 vs 合約 {c['payment_terms_days']} 天"
                  f"（內部自洽，僅跨文件可發現）")


def d_fake_payment(inv, con, led, rng, used) -> Optional[Defect]:
    """標記為已收款，但銀行流水沒有對應入帳 —— 虛報收款美化帳齡。"""
    cands = [i for i in inv if i.get("status") == "PENDING"
             and str(i.get("invoice_number")) not in used]
    if not cands:
        return None
    t = rng.choice(cands)
    used.add(str(t["invoice_number"]))
    t["status"] = "PAID"
    t["paid_date"] = t["due_date"]
    return Defect("D15", 3, "虛報收款", "BANK-01", t["invoice_number"],
                  "標記已收款但銀行流水無對應入帳")


def d_ledger_tamper(inv, con, led, rng, used) -> Optional[Defect]:
    """竄改流水金額但忘記重算後續餘額 —— 最常見的流水造假破綻。"""
    if len(led) < 10:
        return None
    idx = rng.randint(2, len(led) - 3)
    led[idx]["amount"] = float(led[idx]["amount"]) * 3
    return Defect("D16", 3, "銀行流水餘額不連續", "LEDGER-01",
                  f"第{idx+1}筆流水", "改了金額但沒重算後續餘額")


def d_over_commitment(inv, con, led, rng, used) -> Optional[Defect]:
    """單一買方累計開票遠超合約年度承諾額。"""
    if not con:
        return None
    c = rng.choice(con)
    cands = [i for i in inv if i.get("buyer_ban") == c["buyer_ban"]
             and str(i.get("invoice_number")) not in used]
    if not cands:
        return None
    t = rng.choice(cands)
    used.add(str(t["invoice_number"]))
    t["sales_amount"] = int(c["annual_commitment_amount"] * 2.5)
    t["tax_amount"] = round(t["sales_amount"] * 0.05)
    t["total_amount"] = t["sales_amount"] + t["tax_amount"]
    return Defect("D17", 3, "累計開票超出合約承諾額", "CONTRACT-02",
                  t["invoice_number"], "單張即超過年度承諾額 2.5 倍")


# ── Tier 4：統計性 ────────────────────────────────────────────────────

def d_round_numbers(inv, con, led, rng, used) -> Optional[Defect]:
    """把三成發票改成整萬元 —— 人為編造金額的典型痕跡。"""
    cands = [i for i in inv if str(i.get("invoice_number")) not in used]
    n = max(1, int(len(inv) * 0.35))
    for t in rng.sample(cands, min(n, len(cands))):
        t["sales_amount"] = round(t["sales_amount"] / 10_000) * 10_000 or 10_000
        t["tax_amount"] = round(t["sales_amount"] * 0.05)
        t["total_amount"] = t["sales_amount"] + t["tax_amount"]
        used.add(str(t["invoice_number"]))
    return Defect("D18", 4, "整數金額偏好", "FORENSIC-01",
                  f"{n} 張發票", "35% 的銷售額被改為萬元整數")


def d_benford_violation(inv, con, led, rng, used) -> Optional[Defect]:
    """
    違反班佛定律，但**保持跨越的數量級不變**。

    第一版直接把金額改成 randint(400_000, 999_999)（僅 0.4 decades），
    結果被 FORENSIC-02 的適用性前提擋掉（span < 1.5 就不檢定），
    命中率 0% —— 那不是偵測失敗，是**注入器讓檢定失去適用條件**。

    改為在每個數量級內部各自扭曲首位數字分布：
    保持整體跨 2+ decades，但首位數字趨於均勻。
    這才是真正在測「班佛檢定抓不抓得到人為編造」。
    """
    cands = [i for i in inv if str(i.get("invoice_number")) not in used]
    if len(cands) < 50:
        return None
    for t in cands:
        # 在原本的數量級內重抽，首位數字取均勻（真實應為對數分布）
        mag = len(str(int(t["sales_amount"]))) - 1
        first = rng.randint(1, 9)
        rest = rng.randint(0, 10 ** mag - 1) if mag > 0 else 0
        t["sales_amount"] = first * (10 ** mag) + rest
        t["tax_amount"] = round(t["sales_amount"] * 0.05)
        t["total_amount"] = t["sales_amount"] + t["tax_amount"]
        used.add(str(t["invoice_number"]))
    return Defect("D19", 4, "金額分布違反班佛定律", "FORENSIC-02",
                  f"{len(cands)} 張發票",
                  "首位數字改為均勻分布（保持原數量級跨度）")


def d_weekend_burst(inv, con, led, rng, used) -> Optional[Defect]:
    """把大量發票日期改到週末 —— 一次補製的痕跡。"""
    cands = [i for i in inv if str(i.get("invoice_number")) not in used]
    n = max(1, int(len(inv) * 0.30))
    picked = rng.sample(cands, min(n, len(cands)))
    for t in picked:
        d = date.fromisoformat(t["invoice_date"])
        shift = (5 - d.weekday()) % 7
        nd = d + timedelta(days=shift)
        t["invoice_date"] = nd.isoformat()
        t["due_date"] = (nd + timedelta(days=int(t["payment_terms_days"]))).isoformat()
        used.add(str(t["invoice_number"]))
    return Defect("D20", 4, "假日開票異常", "FORENSIC-03",
                  f"{len(picked)} 張發票", "30% 的發票日期被改到週末")


# ── Tier 5：已知抓不到（誠實揭露能力邊界）─────────────────────────────

def d_perfect_forgery(inv, con, led, rng, used) -> Optional[Defect]:
    """
    完全自洽的偽造發票：統編有效、算術正確、日期合理、帳期符合合約、
    甚至銀行流水也偽造了對應入帳。**憑證之間毫無矛盾。**

    這種造假**我們抓不到**，需要物流單、報關單等外部佐證。
    刻意放進評測是為了讓 recall 數字誠實 ——
    一份只統計「抓得到的樣態」的報告，recall 會虛高。
    """
    from .textnorm import validate_tax_id
    src = rng.choice(inv)
    ban = src["buyer_ban"]
    idate = date.fromisoformat(src["invoice_date"])
    while idate.weekday() >= 5:
        idate -= timedelta(days=1)
    sales = int(math.exp(rng.uniform(math.log(30_000), math.log(900_000))))
    tax = round(sales * 0.05)
    num = f"AB{rng.randint(10_000_000, 99_999_999)}"
    terms = int(src["payment_terms_days"])
    fake = {
        "doc_type": "AR_INVOICE", "invoice_number": num,
        "invoice_date": idate.isoformat(),
        "seller_ban": src["seller_ban"], "seller_name": src["seller_name"],
        "buyer_ban": ban, "buyer_name": src["buyer_name"],
        "sales_amount": sales, "tax_amount": tax, "total_amount": sales + tax,
        "payment_terms_days": terms,
        "due_date": (idate + timedelta(days=terms)).isoformat(),
        "status": "PAID", "paid_date": (idate + timedelta(days=terms + 2)).isoformat(),
        "source_note": "synthetic",
    }
    inv.append(fake)
    # 連銀行流水都一起偽造，讓勾稽也對得上
    if led:
        last_bal = float(led[-1].get("balance", 0))
        led.append({"date": fake["paid_date"], "description": f"匯入-{fake['buyer_name']}",
                    "counterparty_ban": ban, "reference": num,
                    "amount": sales + tax, "balance": last_bal + sales + tax})
    assert validate_tax_id(ban)
    return Defect("D21", 5, "完全自洽的偽造發票", None, num,
                  "統編有效、算術正確、日期合理、帳期符合合約、流水也偽造了對應入帳。"
                  "**已知抓不到** —— 需物流單/報關單等外部佐證")


def d_shell_company(inv, con, led, rng, used) -> Optional[Defect]:
    """
    人頭公司：統編**通過檢核碼**（真的去註冊一家空殼），交易也真的走了帳。
    純憑證比對抓不到，需要商工登記的負責人關聯或實地訪查。
    """
    t = _pick(inv, rng, used)
    if not t:
        return None
    from .textnorm import validate_tax_id
    ban = None
    for _ in range(2000):
        cand = str(rng.randint(10_000_000, 99_999_999))
        if validate_tax_id(cand):
            ban = cand
            break
    if not ban:
        return None
    t["buyer_ban"] = ban
    t["buyer_name"] = "宏達貿易有限公司"
    return Defect("D22", 5, "人頭公司買方", None, t["invoice_number"],
                  "統編通過檢核碼、交易走帳。**已知抓不到** —— "
                  "需商工登記負責人關聯或實地訪查")


ALL_DEFECTS: list[tuple[str, int, Callable]] = [
    ("D01", 1, d_invalid_ban), ("D02", 1, d_arith_mismatch),
    ("D03", 1, d_wrong_tax_rate), ("D04", 1, d_future_date),
    ("D05", 1, d_reversed_due), ("D06", 1, d_negative_amount),
    ("D07", 1, d_negative_tax), ("D08", 1, d_absurd_term),
    ("D09", 2, d_self_dealing), ("D10", 2, d_duplicate_number),
    ("D11", 2, d_duplicate_billing), ("D12", 2, d_sequential_numbers),
    ("D13", 2, d_related_party_ban),
    ("D14", 3, d_term_mismatch), ("D15", 3, d_fake_payment),
    ("D16", 3, d_ledger_tamper), ("D17", 3, d_over_commitment),
    ("D18", 4, d_round_numbers), ("D19", 4, d_benford_violation),
    ("D20", 4, d_weekend_burst),
    ("D21", 5, d_perfect_forgery), ("D22", 5, d_shell_company),
]

# 統計性樣態（Tier 4）會大量改動資料，與其他樣態同時注入會互相干擾，
# 所以預設分開評測。
STATISTICAL_DEFECTS = {"D18", "D19", "D20"}


def inject(invoices: list[dict], contracts: list[dict], ledger: list[dict],
           defect_ids: list[str], seed: int = 0) -> InjectionResult:
    """對一份乾淨的資料集注入指定的瑕疵，回傳結果與答案卷。"""
    rng = random.Random(seed)
    inv = [dict(i) for i in invoices]
    con = [dict(c) for c in contracts]
    led = [dict(x) for x in ledger]
    used: set[str] = set()
    defects: list[Defect] = []

    for did, _tier, fn in ALL_DEFECTS:
        if did not in defect_ids:
            continue
        d = fn(inv, con, led, rng, used)
        if d:
            defects.append(d)
    return InjectionResult(inv, con, led, defects)
