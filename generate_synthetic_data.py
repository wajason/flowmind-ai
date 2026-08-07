#!/usr/bin/env python3
"""
generate_synthetic_data.py — 合成中小企業營運資料產生器 v4
=============================================================================
產生一整套「銀行受理應收帳款承購時實際會收到的文件」，用於 demo 與評測。

v4 相對 v3 的四項修正，全部是被 flowmind/crosscheck.py 的決定性驗證當場抓出來的
（一個能抓出自己資料造假的系統，才有資格說自己能抓出客戶資料的問題）：

  1. 統一編號真的通過財政部檢核碼演算法。
     v3 的 docstring 寫著「通過檢查碼規則」，但實作只是隨機 8 碼數字——
     90 張發票有 73 張的統編根本不可能存在。這在 demo 現場被評審拿計算機一算就穿幫。

  2. 導入「客戶主檔」。v3 每開一張發票就重新隨機一組 buyer_ban，
     結果同一個買方名稱底下有十幾個不同統編。這讓買方集中度、合約勾稽、
     信用評等全部失真——而買方集中度正是銀行核應收帳款額度時最先看的指標。

  3. 公司名稱改為合理的台灣中小企業命名。v3 產生「Griffin股份有限公司」，
     一眼就看得出是英文假名產生器，會直接損害提案的可信度。

  4. 收款行為模型化。v3 的規則是「到期後 15% 永遠不付」，跑 24 個月後
     逾期占未收帳款高達 84%——這種體質的公司銀行不會放款，demo 情境自相矛盾。
     v4 改成每個買方有自己的付款習性（準時/偏慢/不穩），逾期是「還沒收到」
     而不是「永遠收不到」，整體逾期率落在中小企業實務常見的區間。

  5. 銀行流水帶入交易對手與發票號碼，讓「發票 ↔ 入帳」真的勾稽得起來。
     沒有這個欄位，「銀行流水佐證」就只是一句口號。

所有企業名稱、統一編號皆為虛構，不對應任何真實企業。
發票欄位命名對齊財政部「電子發票資料交換標準訊息建置指引」B2B 訊息規格。

用法：
  python generate_synthetic_data.py --company "宏昇機械" --industry 精密機械 --seed 42
  python generate_synthetic_data.py --stress          # 決賽 demo 的現金缺口情境
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import date, timedelta
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# 產業設定
# ═══════════════════════════════════════════════════════════════════════════
INDUSTRY_PROFILES = {
    # ── 製造業 ────────────────────────────────────────────────────────
    "精密機械": {"terms": [30, 60, 90], "term_w": [0.3, 0.5, 0.2],
                 "order_range": (80_000, 450_000), "gross_margin": 0.22,
                 "peak_months": [3, 4, 9, 10], "peak_multiplier": 1.4},
    "電子零組件": {"terms": [30, 45, 60], "term_w": [0.4, 0.3, 0.3],
                   "order_range": (60_000, 600_000), "gross_margin": 0.18,
                   "peak_months": [8, 9, 10, 11], "peak_multiplier": 1.6},
    "食品加工": {"terms": [15, 30, 45], "term_w": [0.5, 0.4, 0.1],
                 "order_range": (30_000, 150_000), "gross_margin": 0.28,
                 "peak_months": [1, 2, 9], "peak_multiplier": 1.5},
    "紡織成衣": {"terms": [30, 60, 90], "term_w": [0.35, 0.4, 0.25],
                 "order_range": (40_000, 250_000), "gross_margin": 0.20,
                 "peak_months": [2, 3, 7, 8], "peak_multiplier": 1.35},

    # ── 批發業 ────────────────────────────────────────────────────────
    # 補這三類是因為真實承保統計顯示批發業才是最大宗
    # （機械器具批發 14.36%、食品飲料批發 9.92%、建材批發 7.66%），
    # 而原本的合成器只做製造業，等於漏掉了市場的一半。
    # 批發業的特徵與製造業明顯不同：毛利率低很多（不做加工，賺價差）、
    # 帳期短、單筆金額小但筆數多、交易對手數量多，
    # 這些差異直接影響買方集中度與現金流曲線的形狀。
    "機械器具批發": {"terms": [30, 45, 60], "term_w": [0.45, 0.35, 0.20],
                     "order_range": (50_000, 800_000), "gross_margin": 0.12,
                     "peak_months": [3, 4, 10, 11], "peak_multiplier": 1.3},
    "食品批發": {"terms": [7, 15, 30], "term_w": [0.3, 0.45, 0.25],
                 "order_range": (20_000, 200_000), "gross_margin": 0.09,
                 "peak_months": [1, 2, 9, 12], "peak_multiplier": 1.6},
    "建材批發": {"terms": [30, 60, 90], "term_w": [0.3, 0.45, 0.25],
                 "order_range": (60_000, 900_000), "gross_margin": 0.14,
                 "peak_months": [3, 4, 5, 10], "peak_multiplier": 1.35},
}

# 台灣中小企業常見的命名結構：吉祥字/方位字 + 產業字 + 組織型態
_NAME_HEAD = ["宏", "鴻", "永", "泰", "晶", "昱", "群", "隆", "銓", "皓",
              "德", "偉", "順", "銘", "冠", "祥", "睿", "紘"]
_NAME_TAIL = ["昇", "陽", "達", "宇", "翔", "毅", "鋒", "誠", "陞", "驊", "麒", "澤"]
_BIZ_WORD = {
    "精密機械": ["精密工業", "機械", "工業", "自動化科技", "精機"],
    "電子零組件": ["電子", "科技", "電子科技", "半導體", "光電"],
    "食品加工": ["食品", "食品工業", "生技食品", "食品科技"],
    "紡織成衣": ["紡織", "織品", "實業", "紡織實業"],
    "機械器具批發": ["機械", "工業社", "機電", "貿易", "企業"],
    "食品批發": ["food".replace("food", "食品"), "食品貿易", "商行", "企業"],
    "建材批發": ["建材", "建材企業", "材料", "工程材料"],
}

# 公司名稱後綴的分布 —— 改由 flowmind.calibration 從
# 信保基金真實承保統計取得（見該模組說明）。
# 校準前用的猜測值是 股份 55% / 有限 35% / 企業社 10%，
# 與真實的 股份 48.71% / 有限 47.20% / 獨資合夥 3.74% 明顯不符。
def _suffix_weights() -> dict[str, float]:
    try:
        from flowmind import calibration
        w = calibration.org_suffix_weights()
        if w:
            return w
    except Exception:                                  # noqa: BLE001
        pass
    return {"股份有限公司": 0.55, "有限公司": 0.35, "企業社": 0.10}

# 買方的付款習性。這三種在中小企業供應鏈裡都真實存在，
# 而且銀行的授信人員正是靠這個分布在判斷「這批應收到底值多少錢」。
# skip_prob 是「最終變成呆帳」的機率，不是「這期沒付」的機率。
# 加權後的整體呆帳率約 1.9%，落在製造業中小企業常見的 1~3% 區間；
# 設太高會讓這家公司在銀行眼中直接出局，demo 就失去說服力。
PAYER_BEHAVIOURS = {
    "準時":   {"delay_mean": 2,  "delay_sd": 3,  "skip_prob": 0.005, "weight": 0.45},
    "偏慢":   {"delay_mean": 18, "delay_sd": 10, "skip_prob": 0.020, "weight": 0.40},
    "不穩定": {"delay_mean": 40, "delay_sd": 25, "skip_prob": 0.060, "weight": 0.15},
}


# ═══════════════════════════════════════════════════════════════════════════
# 統一編號：真的通過財政部檢核碼
# ═══════════════════════════════════════════════════════════════════════════
_BAN_WEIGHTS = [1, 2, 1, 2, 1, 2, 4, 1]


def _ban_checksum_ok(ban: str) -> bool:
    total = 0
    for digit, w in zip((int(c) for c in ban), _BAN_WEIGHTS):
        p = digit * w
        total += p // 10 + p % 10
    if total % 5 == 0:
        return True
    return ban[6] == "7" and (total + 1) % 5 == 0


def make_valid_ban(rng: random.Random) -> str:
    """
    產生通過檢核碼的虛構統編。做法是隨機前 7 碼、暴力搜尋末碼，
    找不到就重抽——比反推公式短，而且不會因為公式寫錯而產出一堆無效號碼。
    刻意避開 0 開頭（真實統編不會以 0 起始）。
    """
    while True:
        head = str(rng.randint(1, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(6))
        for last in range(10):
            ban = head + str(last)
            if _ban_checksum_ok(ban):
                return ban


def make_company_name(rng: random.Random, industry: str) -> str:
    biz = rng.choice(_BIZ_WORD.get(industry, ["實業"]))
    stem = rng.choice(_NAME_HEAD) + rng.choice(_NAME_TAIL)
    w = _suffix_weights()
    suffix = rng.choices(list(w), weights=list(w.values()))[0]
    return f"{stem}{biz}{suffix}"


# ═══════════════════════════════════════════════════════════════════════════
# 產生器
# ═══════════════════════════════════════════════════════════════════════════
class SyntheticSMEGenerator:
    def __init__(self, company_name: str, industry: str, months: int, seed: int | None = None):
        if industry not in INDUSTRY_PROFILES:
            raise ValueError(f"industry 必須是 {list(INDUSTRY_PROFILES)} 其中之一")
        self.rng = random.Random(seed)
        self.company_name = company_name
        self.industry = industry
        self.profile = INDUSTRY_PROFILES[industry]
        self.seller_ban = make_valid_ban(self.rng)
        self.months = months
        self.today = date.today()
        self.start_date = self.today - timedelta(days=30 * months)

        self.customers = self._build_customer_master()
        self.suppliers = self._build_supplier_master()
        self.contracts = self._build_contracts()

    # ── 主檔 ───────────────────────────────────────────────────────────────
    def _build_customer_master(self, n: int = 9) -> list[dict]:
        """
        買方主檔。每個買方有固定統編、固定約定帳期、固定付款習性。
        份額用遞減權重分配，讓「前二大客戶占四成上下」——
        這是中小企業真實的客戶結構，均勻分布反而不像真的。
        """
        behaviours = list(PAYER_BEHAVIOURS)
        b_weights = [PAYER_BEHAVIOURS[b]["weight"] for b in behaviours]
        customers = []
        for i in range(n):
            customers.append({
                "ban": make_valid_ban(self.rng),
                "name": make_company_name(self.rng, self.industry),
                "payment_terms_days": self.rng.choices(
                    self.profile["terms"], weights=self.profile["term_w"])[0],
                "behaviour": self.rng.choices(behaviours, weights=b_weights)[0],
                # 訂單份額遞減：第一大客戶約為第九大的 4 倍
                "share_weight": 1.0 / (i + 1) ** 0.8,
            })
        return customers

    def _build_supplier_master(self, n: int = 6) -> list[dict]:
        kinds = ["原料", "五金", "包材", "零件", "模具", "表面處理"]
        return [{
            "ban": make_valid_ban(self.rng),
            "name": f"{self.rng.choice(_NAME_HEAD)}{self.rng.choice(kinds)}"
                    f"{self.rng.choices(['有限公司', '股份有限公司'], weights=[0.7, 0.3])[0]}",
            "payment_terms_days": self.rng.choice([15, 30, 45]),
        } for _ in range(n)]

    def _build_contracts(self) -> list[dict]:
        """
        只有前四大客戶簽有年度基本買賣合約——這符合實務：
        小額零星客戶通常只有訂單沒有合約，而銀行受理應收帳款承購時，
        「有沒有合約」正是決定該筆應收能不能承作的關鍵。
        """
        out = []
        for i, c in enumerate(self.customers[:4]):
            start = self.start_date + timedelta(days=self.rng.randint(0, 60))
            out.append({
                "doc_type": "SALES_CONTRACT",
                "contract_number": f"CT-{start.year}-{i+1:03d}",
                "seller_ban": self.seller_ban,
                "seller_name": self.company_name,
                "buyer_ban": c["ban"],
                "buyer_name": c["name"],
                "effective_date": start.isoformat(),
                "expiry_date": (start + timedelta(days=730)).isoformat(),
                "payment_terms_days": c["payment_terms_days"],
                "annual_commitment_amount": self.rng.randint(6, 30) * 1_000_000,
                "currency": "TWD",
                "recourse_note": "本合約未約定禁止債權讓與，得作為應收帳款承購標的",
                "source_note": "synthetic - 年度基本買賣合約，供發票帳期勾稽使用",
            })
        return out

    # ── 應收帳款 ───────────────────────────────────────────────────────────
    def _month_weight(self, d: date) -> float:
        return self.profile["peak_multiplier"] if d.month in self.profile["peak_months"] else 1.0

    def generate_receivables(self, n_invoices: int) -> list[dict]:
        rng = self.rng
        weights = [c["share_weight"] for c in self.customers]
        invoices = []
        used_numbers: set[str] = set()

        for _ in range(n_invoices):
            cust = rng.choices(self.customers, weights=weights)[0]
            issue_date = self._random_date_in_range()
            term_days = cust["payment_terms_days"]
            due_date = issue_date + timedelta(days=term_days)

            base = rng.randint(*self.profile["order_range"])
            sales_amount = round(base * self._month_weight(issue_date))
            tax_amount = round(sales_amount * 0.05)

            num = f"AB{rng.randint(10_000_000, 99_999_999)}"
            while num in used_numbers:
                num = f"AB{rng.randint(10_000_000, 99_999_999)}"
            used_numbers.add(num)

            # ── 收款行為：依買方習性決定實際入帳日 ──────────────────────
            beh = PAYER_BEHAVIOURS[cust["behaviour"]]
            delay = max(-3, round(rng.gauss(beh["delay_mean"], beh["delay_sd"])))
            will_default = rng.random() < beh["skip_prob"]
            paid_date = None if will_default else due_date + timedelta(days=delay)

            days_past_due = (self.today - due_date).days
            if paid_date is not None and paid_date <= self.today:
                status = "PAID"
            elif will_default and days_past_due > 180:
                # 逾期超過 180 天且無收款跡象 → 轉呆帳沖銷。
                # 這不是為了讓數字好看，而是實際的會計慣例（商業會計法備抵呆帳）：
                # 沒有企業會把三年前收不到的帳一直掛在應收帳款裡。
                # 若不做這一步，逾期款會隨時間無限累積，跑 24 個月後
                # 逾期占比會衝到 50% 以上，變成一家銀行根本不會放款的公司，
                # 整個 demo 情境就自相矛盾了。
                status = "WRITTEN_OFF"
            elif due_date < self.today:
                # 「還沒收到」而不是「收不到」：中小企業的逾期多半是慢收，
                # 最終仍會入帳。這個區分直接影響現金流預測的可信度。
                status = "OVERDUE"
            else:
                status = "PENDING"

            invoices.append({
                "doc_type": "AR_INVOICE",
                "invoice_number": num,
                "invoice_date": issue_date.isoformat(),
                "seller_ban": self.seller_ban,
                "seller_name": self.company_name,
                "buyer_ban": cust["ban"],
                "buyer_name": cust["name"],
                "sales_amount": sales_amount,
                "tax_amount": tax_amount,
                "total_amount": sales_amount + tax_amount,
                "payment_terms_days": term_days,
                "due_date": due_date.isoformat(),
                "status": status,
                "paid_date": paid_date.isoformat() if status == "PAID" else None,
                "buyer_payment_behaviour": cust["behaviour"],
                "source_note": "synthetic - schema aligned to 財政部電子發票 B2B 訊息規格 "
                               "(InvoiceNumber/SellerBAN/BuyerBAN/SalesAmount/TaxAmount/TotalAmount)",
            })
        return invoices

    # ── 應付帳款 ───────────────────────────────────────────────────────────
    def generate_payables(self, n_bills: int) -> list[dict]:
        rng = self.rng
        bills = []
        for _ in range(n_bills):
            sup = rng.choice(self.suppliers)
            issue_date = self._random_date_in_range()
            term_days = sup["payment_terms_days"]
            due_date = issue_date + timedelta(days=term_days)
            amount = round(rng.randint(*self.profile["order_range"])
                           * (1 - self.profile["gross_margin"]) * rng.uniform(0.3, 0.6))
            bills.append({
                "doc_type": "AP_BILL",
                "bill_number": f"AP{rng.randint(10_000_000, 99_999_999)}",
                "issue_date": issue_date.isoformat(),
                "buyer_ban": self.seller_ban,
                "buyer_name": self.company_name,
                "supplier_ban": sup["ban"],
                "supplier_name": sup["name"],
                "amount": amount,
                "payment_terms_days": term_days,
                "due_date": due_date.isoformat(),
                # 自家對供應商一律準時付款：中小企業若對上游延票，
                # 供應商很快就會改要求現金交易，這是實務上的硬約束。
                "status": "PAID" if due_date < self.today else "PENDING",
                "source_note": "synthetic - 應付帳款，供現金流缺口計算使用",
            })
        return bills

    # ── 銀行流水 ───────────────────────────────────────────────────────────
    def generate_bank_ledger(self, receivables, payables, opening_balance: int):
        """
        帶入交易對手統編與憑證號碼，讓「發票 ↔ 入帳」可以真的勾稽。
        真實銀行對帳單的摘要欄常常只有匯款人簡稱，這裡刻意保留完整資訊，
        因為我們要驗證的是勾稽邏輯本身，OCR/摘要解析是另一層的問題。
        """
        events = []
        for r in receivables:
            if r["status"] == "PAID" and r.get("paid_date"):
                events.append((r["paid_date"], r["total_amount"],
                               f"匯入-{r['buyer_name']}", r["buyer_ban"], r["invoice_number"]))
        for p in payables:
            if p["status"] == "PAID":
                events.append((p["due_date"], -p["amount"],
                               f"匯出-{p['supplier_name']}", p["supplier_ban"], p["bill_number"]))
        for oe in self.generate_opex_entries()[0]:
            events.append((oe["date"], oe["amount"], oe["description"], "", ""))

        events.sort(key=lambda x: x[0])
        rows, balance = [], opening_balance
        for d, amt, desc, ban, ref in events:
            balance += amt
            rows.append({"date": d, "description": desc, "counterparty_ban": ban,
                         "reference": ref, "amount": amt, "balance": balance})
        return rows, balance

    def _random_date_in_range(self) -> date:
        span = (self.today - self.start_date).days
        return self.start_date + timedelta(days=self.rng.randint(0, span))

    def monthly_fixed_opex(self) -> int:
        avg_order = sum(self.profile["order_range"]) / 2
        return round(avg_order * 0.9)

    def generate_opex_entries(self):
        entries, opex = [], self.monthly_fixed_opex()
        cursor = date(self.start_date.year, self.start_date.month, 5)
        while cursor <= self.today:
            entries.append({"date": cursor.isoformat(), "amount": -opex,
                            "description": "固定營運支出(人事/房租/水電)"})
            cursor = (cursor.replace(day=1) + timedelta(days=32)).replace(day=5)
        return entries, opex


# ═══════════════════════════════════════════════════════════════════════════
# 決定性現金流缺口計算（不經 LLM）
# ═══════════════════════════════════════════════════════════════════════════
def compute_cash_flow_projection(receivables, payables, current_balance: int,
                                 horizon_days: int = 90, monthly_opex: int = 0) -> dict:
    """
    把未來到期的應收/應付攤在時間軸上，找出最早出現負餘額的日期與金額。

    逾期應收「不」計入未來現金流：收不收得回來不確定，
    樂觀假設它會準時入帳會嚴重高估償債能力——這正是銀行徵信最不能接受的失真。
    改為獨立列為 overdue_receivables_total 風險指標。
    """
    today = date.today()
    horizon = today + timedelta(days=horizon_days)
    events = []

    if monthly_opex:
        cursor = date(today.year, today.month, 5)
        while cursor <= horizon:
            if cursor >= today:
                events.append((cursor, -monthly_opex, "outflow", "固定營運支出(人事/房租/水電)", ""))
            cursor = (cursor.replace(day=1) + timedelta(days=32)).replace(day=5)

    overdue_total = 0
    for r in receivables:
        if r["status"] == "PENDING":
            d = date.fromisoformat(r["due_date"])
            if today <= d <= horizon:
                events.append((d, r["total_amount"], "inflow", r["buyer_name"], r["invoice_number"]))
        elif r["status"] == "OVERDUE":
            overdue_total += r["total_amount"]

    for p in payables:
        if p["status"] == "PENDING":
            d = date.fromisoformat(p["due_date"])
            if today <= d <= horizon:
                events.append((d, -p["amount"], "outflow", p["supplier_name"], p["bill_number"]))

    events.sort(key=lambda x: x[0])
    running, timeline = current_balance, []
    gap_date = gap_amount = None
    for d, amt, kind, party, ref in events:
        running += amt
        timeline.append({"date": d.isoformat(), "amount": amt, "type": kind,
                         "counterparty": party, "reference": ref,
                         "projected_balance": running})
        if running < 0 and gap_date is None:
            gap_date, gap_amount = d.isoformat(), running

    return {
        "current_balance": current_balance,
        "horizon_days": horizon_days,
        "timeline": timeline,
        "gap_detected": gap_date is not None,
        "gap_date": gap_date,
        "gap_amount": gap_amount,
        "overdue_receivables_total": overdue_total,
        "computation_method": "deterministic_netting",
    }


# ═══════════════════════════════════════════════════════════════════════════
# 負向對照組：注入已知缺陷
# ═══════════════════════════════════════════════════════════════════════════
def inject_known_defects(invoices: list[dict], gen: "SyntheticSMEGenerator") -> list[dict]:
    """
    注入五種真實世界確實發生過的憑證瑕疵，並回傳一份「答案卷」。

    這五種不是隨便編的，都對應應收帳款融資實務上銀行真正在防的事：
      1. 自我交易      —— 關係企業互開發票虛增營收去換額度
      2. 重複請款      —— 同一批貨開兩張票分別向兩家銀行融資
      3. 統編無效      —— 買方根本不存在（人頭公司）
      4. 金額不符      —— 銷售額＋稅額 ≠ 總額，常見於竄改後忘了改總計
      5. 帳期與合約不符 —— 對銀行報 30 天、實際合約 90 天，美化週轉天數

    回傳的答案卷用於自動比對：引擎必須「剛好」抓到這五項，
    既不能漏（漏了代表防線無效），也不該在乾淨資料上誤報（誤報會讓人不再看報表）。
    """
    rng = gen.rng
    answers: list[dict] = []
    picks = rng.sample(range(len(invoices)), 5)

    # 1. 自我交易
    inv = invoices[picks[0]]
    inv["buyer_ban"] = inv["seller_ban"]
    inv["buyer_name"] = inv["seller_name"]
    answers.append({"check_id": "FRAUD-01", "invoice_number": inv["invoice_number"],
                    "defect": "買方統編與賣方相同（自我交易）"})

    # 2. 重複請款：複製一張票，換號碼但買方/金額/日期完全相同
    src = invoices[picks[1]]
    dup = dict(src)
    dup["invoice_number"] = f"AB{rng.randint(10_000_000, 99_999_999)}"
    invoices.append(dup)
    answers.append({"check_id": "DUP-02", "invoice_number": dup["invoice_number"],
                    "defect": f"與 {src['invoice_number']} 同買方/同金額/同日期"})

    # 3. 統編檢核碼失敗：把末碼 +1，必然破壞檢核
    inv = invoices[picks[2]]
    inv["buyer_ban"] = inv["buyer_ban"][:7] + str((int(inv["buyer_ban"][7]) + 1) % 10)
    answers.append({"check_id": "TAXID-01", "invoice_number": inv["invoice_number"],
                    "defect": f"買方統編 {inv['buyer_ban']} 未通過檢核碼"})

    # 4. 金額加總不符
    inv = invoices[picks[3]]
    inv["total_amount"] = inv["total_amount"] + 1000
    answers.append({"check_id": "ARITH-01", "invoice_number": inv["invoice_number"],
                    "defect": "銷售額＋稅額 ≠ 總額（差 1,000 元）"})

    # 5. 帳期與合約不符（挑一張有合約的買方）
    contracted = {c["buyer_ban"]: c for c in gen.contracts}
    for i in invoices:
        if i["buyer_ban"] in contracted and i["invoice_number"] not in \
                {a["invoice_number"] for a in answers}:
            new_terms = 30 if contracted[i["buyer_ban"]]["payment_terms_days"] != 30 else 90
            i["payment_terms_days"] = new_terms
            # 到期日一併改成一致 —— 真的要美化週轉天數的人不會只改一個欄位，
            # 這樣才能逼引擎必須靠「跨文件比對合約」抓，而不是靠單張發票的內部矛盾。
            i["due_date"] = (date.fromisoformat(i["invoice_date"])
                             + timedelta(days=new_terms)).isoformat()
            answers.append({"check_id": "TERM-02", "invoice_number": i["invoice_number"],
                            "defect": "發票帳期與年度合約約定不符（內部欄位自洽，僅跨文件比對可發現）"})
            break
    return answers


# ═══════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description="產生合成中小企業營運資料集")
    p.add_argument("--company", default="宏昇機械股份有限公司")
    p.add_argument("--industry", default="精密機械", choices=list(INDUSTRY_PROFILES))
    p.add_argument("--months", type=int, default=24,
                   help="歷史期間月數，預設 24 個月以貼近銀行徵信常見審核期間")
    p.add_argument("--n-invoices", type=int, default=90)
    p.add_argument("--n-bills", type=int, default=70)
    p.add_argument("--opening-balance", type=int, default=850_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", default="./data/raw/CASE-0001")
    p.add_argument("--inject-defects", action="store_true",
                   help="刻意注入五種已知的憑證瑕疵，作為交叉驗證引擎的負向對照組。"
                        "一套永遠回報『全部通過』的檢查，本身不構成任何證據——"
                        "必須證明它抓得到問題，『它說沒問題』才有意義")
    p.add_argument("--stress", action="store_true",
                   help="注入一筆即將到期的大額原料款，模擬「接獲大單需先墊料」情境。"
                        "這是供應鏈金融產品標準的情境壓力測試，輸出會明確標註為壓力測試資料")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gen = SyntheticSMEGenerator(args.company, args.industry, args.months, seed=args.seed)
    receivables = gen.generate_receivables(args.n_invoices)
    payables = gen.generate_payables(args.n_bills)
    bank_rows, ending_balance = gen.generate_bank_ledger(receivables, payables, args.opening_balance)
    monthly_opex = gen.monthly_fixed_opex()

    if args.stress:
        due = date.today() + timedelta(days=gen.rng.randint(18, 35))
        sup = gen.suppliers[0]
        amount = round((ending_balance + monthly_opex) * gen.rng.uniform(1.15, 1.45))
        payables.append({
            "doc_type": "AP_BILL", "bill_number": "AP-STRESS-01",
            "issue_date": date.today().isoformat(),
            "buyer_ban": gen.seller_ban, "buyer_name": gen.company_name,
            "supplier_ban": sup["ban"], "supplier_name": sup["name"],
            "amount": amount, "payment_terms_days": (due - date.today()).days,
            "due_date": due.isoformat(), "status": "PENDING",
            "source_note": "STRESS TEST SCENARIO — 模擬接獲大單後的原料預付款，非常態資料。"
                           "金額校準為超過目前銀行餘額＋當期營運支出，以確保情境確實觸發缺口",
        })

    injected: list[dict] = []
    if args.inject_defects:
        injected = inject_known_defects(receivables, gen)

    projection = compute_cash_flow_projection(receivables, payables, ending_balance,
                                              monthly_opex=monthly_opex)

    def dump(name, obj):
        (outdir / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    dump("receivables.json", receivables)
    dump("payables.json", payables)
    dump("contracts.json", gen.contracts)
    dump("cash_flow_projection.json", projection)
    dump("customer_master.json", gen.customers)
    if injected:
        dump("_injected_defects_answer_key.json", injected)

    with open(outdir / "bank_ledger.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["date", "description", "counterparty_ban",
                                          "reference", "amount", "balance"])
        w.writeheader()
        w.writerows(bank_rows)

    # ── 摘要 ──────────────────────────────────────────────────────────────
    open_inv = [r for r in receivables if r["status"] in ("PENDING", "OVERDUE")]
    overdue = [r for r in receivables if r["status"] == "OVERDUE"]
    written_off = [r for r in receivables if r["status"] == "WRITTEN_OFF"]
    open_total = sum(r["total_amount"] for r in open_inv) or 1
    by_buyer: dict[str, int] = {}
    for r in receivables:
        by_buyer[r["buyer_name"]] = by_buyer.get(r["buyer_name"], 0) + r["total_amount"]
    top_name, top_amt = max(by_buyer.items(), key=lambda kv: kv[1])
    grand = sum(by_buyer.values())

    # 校準狀態要講清楚：使用者必須知道拿到的是「對齊真實統計的分布」
    # 還是「開發者拍腦袋的猜測值」。這兩者的可信度差很多。
    try:
        from flowmind import calibration
        if calibration.is_calibrated():
            ind = calibration.industry_distribution()
            print(f"📊 分布已對齊信保基金真實承保統計（{ind.period}）"
                  f"｜組織型態 {', '.join(f'{k} {v:.1%}' for k, v in _suffix_weights().items())}")
        else:
            print("⚠️  找不到承保統計，公司型態分布使用未校準的猜測值")
    except Exception:                                  # noqa: BLE001
        print("⚠️  校準模組不可用，使用未校準的猜測值")

    print(f"公司：{args.company}（{args.industry}）｜期間 {args.months} 個月｜賣方統編 {gen.seller_ban}")
    print(f"客戶主檔 {len(gen.customers)} 家、供應商 {len(gen.suppliers)} 家、年度合約 {len(gen.contracts)} 份")
    print(f"應收 {len(receivables)} 張（未收 {len(open_inv)} 張）／應付 {len(payables)} 筆")
    print(f"逾期 {len(overdue)} 張，占未收帳款 {sum(r['total_amount'] for r in overdue)/open_total:.1%}"
          f"｜呆帳沖銷 {len(written_off)} 張（逾期逾 180 天轉列）")
    print(f"最大買方「{top_name}」占營收 {top_amt/grand:.1%}")
    print(f"每月固定營運支出約 {monthly_opex:,} 元｜銀行流水 {len(bank_rows)} 筆，期末餘額 {ending_balance:,} 元")
    print("--- 決定性現金流缺口偵測 ---")
    if projection["gap_detected"]:
        print(f"⚠ 預估 {projection['gap_date']} 出現缺口，金額約 {abs(projection['gap_amount']):,} 元")
    else:
        print("未來 90 天內未偵測到現金缺口")
    if injected:
        print(f"\n⚠ 已注入 {len(injected)} 項已知缺陷作為負向對照組，答案卷："
              f"_injected_defects_answer_key.json")
        for a in injected:
            print(f"   · [{a['check_id']}] {a['invoice_number']} — {a['defect']}")
    print(f"\n輸出：{outdir.resolve()}")


if __name__ == "__main__":
    main()
