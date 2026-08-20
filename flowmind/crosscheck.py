"""
flowmind.crosscheck — 決定性交叉驗證引擎（輕量知識圖譜的替代方案）
=============================================================================
【這一層完全不使用語言模型，一行 LLM 呼叫都沒有。】

原本規劃裡「知識圖譜驗證」需要 Neo4j 與圖遍歷推理，兩週內做不完。
但回頭看它真正要達成的效果 ——「讓不同來源的證據互相驗證」——
其實不需要圖資料庫：發票、合約、銀行流水之間本來就有共享欄位
（統一編號、金額、日期、發票號碼），把這些欄位串起來比對就夠了。

更重要的是：這些比對全部是可以用純算術確定對錯的事實。
一張發票的稅額是不是 5%、買賣雙方統編是不是同一家（自我交易）、
同一個發票號碼有沒有被開兩次 —— 這些都不該交給語言模型「判斷」。
凡是有明確規則的，就用程式算；語言模型只負責它真正擅長的自然語言理解。

這也是為什麼這一層的結果可以直接呈給銀行授信人員：
每一項都能被對方用同一套規則重算一次，得到完全相同的答案。
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Iterable, Optional

from .textnorm import normalize_tax_id, validate_tax_id

TAX_RATE = 0.05          # 台灣營業稅率
AMOUNT_TOLERANCE = 1.0   # 元。稅額四捨五入的合理誤差，超過就是真的對不上
DATE_TOLERANCE_DAYS = 5  # 銀行入帳與應收到期日的合理落差


class Severity(str, Enum):
    CRITICAL = "critical"   # 足以讓銀行退件或懷疑造假
    WARNING = "warning"     # 需要補件或說明
    INFO = "info"           # 供參考的觀察


@dataclass
class Finding:
    check_id: str
    title: str
    severity: Severity
    passed: bool
    detail: str
    refs: list[str] = field(default_factory=list)   # 涉及的憑證編號，供人工回查

    def as_dict(self) -> dict:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d


def _d(s: Any) -> Optional[date]:
    if isinstance(s, date):
        return s
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except Exception:                                # noqa: BLE001
        return None


# ══════════════════════════════════════════════════════════════════════════
# 逐項檢查
# ══════════════════════════════════════════════════════════════════════════

def check_tax_ids(invoices: list[dict]) -> list[Finding]:
    """統一編號檢核碼。這是財政部公告的算術規則，0.1 毫秒就能算出真偽。"""
    out = []
    bad: list[str] = []          # 人看的說明文字，帶欄位名方便判讀
    bad_refs: list[str] = []     # 純發票號碼，給證據回查用——不能夾帶附註文字
    for inv in invoices:
        for role in ("seller_ban", "buyer_ban"):
            ban = inv.get(role)
            if ban and not validate_tax_id(ban):
                bad.append(f"{inv.get('invoice_number')}({role}={ban})")
                bad_refs.append(str(inv.get("invoice_number")))
    out.append(Finding(
        "TAXID-01", "統一編號檢核碼", Severity.CRITICAL, not bad,
        "全部統編通過財政部檢核碼演算法。" if not bad else
        f"有 {len(bad)} 筆統編未通過檢核碼，代表號碼本身不可能存在："
        + "、".join(bad[:5]) + ("…" if len(bad) > 5 else ""),
        refs=bad_refs[:20],
    ))
    return out


def check_invoice_arithmetic(invoices: list[dict]) -> list[Finding]:
    """銷售額 + 稅額 = 總額，且稅額 ≈ 銷售額 × 5%。"""
    sum_bad, rate_bad = [], []
    for inv in invoices:
        s = inv.get("sales_amount")
        t = inv.get("tax_amount")
        tot = inv.get("total_amount")
        if None in (s, t, tot):
            continue
        if abs((s + t) - tot) > AMOUNT_TOLERANCE:
            sum_bad.append(str(inv.get("invoice_number")))
        # 容差放寬到 1.5 元：發票稅額的進位規則各家系統略有差異，
        # 抓太緊會把正常發票全報成異常，那個報表沒人會看。
        if abs(t - round(s * TAX_RATE)) > 1.5:
            rate_bad.append(str(inv.get("invoice_number")))
    return [
        Finding("ARITH-01", "發票金額加總一致性", Severity.CRITICAL, not sum_bad,
                "所有發票的銷售額＋稅額＝總額。" if not sum_bad else
                f"{len(sum_bad)} 張發票加總不符：" + "、".join(sum_bad[:5]),
                refs=sum_bad[:20]),
        Finding("ARITH-02", "營業稅率合理性(5%)", Severity.WARNING, not rate_bad,
                "稅額皆符合 5% 營業稅。" if not rate_bad else
                f"{len(rate_bad)} 張發票稅額偏離 5%，可能為零稅率/免稅或輸入錯誤，需說明："
                + "、".join(rate_bad[:5]),
                refs=rate_bad[:20]),
    ]


def check_self_dealing(invoices: list[dict]) -> list[Finding]:
    """
    買賣雙方統編相同 = 自己開發票給自己。
    這是應收帳款融資最典型的造假手法（虛增營收去換額度），
    銀行的授信人員一定會查，我們必須在送件前就自己抓出來。
    """
    hits = [str(i.get("invoice_number")) for i in invoices
            if i.get("seller_ban") and
            normalize_tax_id(i.get("seller_ban")) == normalize_tax_id(i.get("buyer_ban"))]
    return [Finding(
        "FRAUD-01", "自我交易偵測", Severity.CRITICAL, not hits,
        "未發現買賣方統編相同的發票。" if not hits else
        f"發現 {len(hits)} 張發票的買方與賣方統編相同（自我交易），"
        f"此類發票不得作為應收帳款融資標的：" + "、".join(hits[:5]),
        refs=hits[:20])]


def check_duplicates(invoices: list[dict]) -> list[Finding]:
    """
    重複開票偵測。分兩種：
      (a) 同一發票號碼出現兩次 —— 通常是系統匯出錯誤
      (b) 同買方、同金額、同日期但號碼不同 —— 才是需要警覺的重複請款
    """
    num_dup = [n for n, c in Counter(
        str(i.get("invoice_number")) for i in invoices).items() if c > 1]

    sig = Counter((i.get("buyer_ban"), i.get("total_amount"), i.get("invoice_date"))
                  for i in invoices)
    near_dup = [s for s, c in sig.items() if c > 1 and all(x is not None for x in s)]

    return [
        Finding("DUP-01", "發票號碼唯一性", Severity.CRITICAL, not num_dup,
                "發票號碼無重複。" if not num_dup else
                f"發現重複發票號碼 {len(num_dup)} 組：" + "、".join(num_dup[:5]),
                refs=num_dup[:20]),
        Finding("DUP-02", "疑似重複請款", Severity.WARNING, not near_dup,
                "未發現同買方、同金額、同日期的可疑重複。" if not near_dup else
                f"有 {len(near_dup)} 組發票的買方/金額/日期完全相同，"
                f"雖然號碼不同，仍建議附出貨單佐證："
                + "、".join(f"{b}/{a}/{d}" for b, a, d in near_dup[:3])),
    ]


def check_terms_consistency(invoices: list[dict], contracts: list[dict]) -> list[Finding]:
    """
    發票的付款帳期是否與合約約定一致，以及到期日是否算對。
    銀行核額度時看的是「合約怎麼約定」，不是「發票上寫幾天」；
    兩者不一致代表企業實際收款條件比合約差，是真實存在的授信風險訊號。
    """
    by_buyer = {normalize_tax_id(c.get("buyer_ban")): c for c in contracts
                if c.get("buyer_ban")}
    term_mismatch, date_mismatch = [], []

    for inv in invoices:
        idate, ddate = _d(inv.get("invoice_date")), _d(inv.get("due_date"))
        terms = inv.get("payment_terms_days")
        if idate and ddate and terms is not None:
            if abs((ddate - idate).days - int(terms)) > 1:
                date_mismatch.append(str(inv.get("invoice_number")))
        c = by_buyer.get(normalize_tax_id(inv.get("buyer_ban")))
        if c and terms is not None and c.get("payment_terms_days") is not None:
            if int(terms) != int(c["payment_terms_days"]):
                term_mismatch.append(
                    f"{inv.get('invoice_number')}(發票{terms}天/合約{c['payment_terms_days']}天)")

    findings = [Finding(
        "TERM-01", "到期日與帳期一致性", Severity.WARNING, not date_mismatch,
        "所有發票到期日 = 開立日 + 約定帳期。" if not date_mismatch else
        f"{len(date_mismatch)} 張發票到期日與帳期天數對不上：" + "、".join(date_mismatch[:5]),
        refs=date_mismatch[:20])]

    if contracts:
        findings.append(Finding(
            "TERM-02", "發票帳期 vs 合約帳期", Severity.WARNING, not term_mismatch,
            "發票帳期與合約約定一致。" if not term_mismatch else
            f"{len(term_mismatch)} 張發票的帳期與合約不符：" + "、".join(term_mismatch[:5]),
            refs=term_mismatch[:20]))
    else:
        findings.append(Finding(
            "TERM-02", "發票帳期 vs 合約帳期", Severity.INFO, True,
            "本案未提供買賣合約，無法比對合約帳期。"
            "銀行受理應收帳款承購時通常會要求合約或訂單，建議補件。"))

    # ── TERM-03：發票開立日不得早於合約生效日 ──────────────────────────
    #
    # 這條檢查是**壓力測試逼出來的**。
    # fraud_injector 的 D24 樣態（把發票日期改到合約簽署之前）注入後，
    # 26 條檢查沒有任何一條被觸發 —— 每個欄位單獨看都完全正常：
    # 日期格式對、不是未來日期、到期日 = 開立日 + 帳期。
    # 只有把發票與合約放在一起看才會發現「這批貨在合約還沒簽時就出了」。
    #
    # 這正是產品的核心主張（跨文件比對才抓得到）少掉的一塊，
    # 而它是被自己的壓力測試找出來的，不是被客戶找出來的。
    early = []
    for inv in invoices:
        idate = _d(inv.get("invoice_date"))
        c = by_buyer.get(normalize_tax_id(inv.get("buyer_ban")))
        if not (idate and c):
            continue
        eff = _d(c.get("effective_date")) or _d(c.get("start_date"))
        if eff and idate < eff:
            early.append(f"{inv.get('invoice_number')}"
                         f"(發票{idate}/合約生效{eff})")
    if contracts:
        # 嚴重度是 WARNING 而不是 CRITICAL，理由必須寫清楚：
        # 真實世界的合約會續約，早於**本份**合約的發票，
        # 可能是在前一份合約下開立的，而我們手上沒有那份。
        # 把「需要補件說明」判成「造假」，會讓使用者不信任所有警示。
        findings.append(Finding(
            "TERM-03", "發票開立日 vs 合約生效日", Severity.WARNING, not early,
            "所有發票的開立日都在合約生效之後。" if not early else
            f"{len(early)} 張發票早於合約生效日，需補前一份合約或訂單佐證："
            + "、".join(early[:5]),
            refs=early[:20]))
    return findings


def check_bank_reconciliation(invoices: list[dict], ledger: list[dict]) -> list[Finding]:
    """
    把「已收款」的發票與銀行流水的實際入帳配對。

    這是整套驗證裡對銀行最有說服力的一項：
    企業自己說收到錢了不算數，要有銀行流水對得上才算數。
    配對條件是金額相符且入帳日落在到期日前後容差內。
    """
    if not ledger:
        return [Finding("BANK-01", "收款與銀行流水勾稽", Severity.INFO, True,
                        "本案未提供銀行流水，無法勾稽實際收款。"
                        "此為授信必要文件，建議補件。")]

    inflows = [l for l in ledger if float(l.get("amount", 0)) > 0]
    unmatched: list[str] = []
    used: set[int] = set()
    by_ref = 0

    # 兩段式配對，順序刻意與真實對帳作業一致：
    #   第一輪：憑證號碼直接對上（銀行匯款附言/虛擬帳號帶回的發票號碼）
    #   第二輪：金額 + 日期區間模糊配對（沒有附言時的退路）
    # 先做強配對再做弱配對，可以避免弱配對先把金額相同的流水搶走，
    # 導致真正有號碼可對的那張反而配不到 —— 這是對帳系統很典型的錯誤。
    for inv in invoices:
        if str(inv.get("status", "")).upper() != "PAID":
            continue
        num = str(inv.get("invoice_number") or "")
        for k, l in enumerate(inflows):
            if k in used:
                continue
            if num and num == str(l.get("reference") or ""):
                used.add(k)
                inv["_matched"] = True
                by_ref += 1
                break

    for inv in invoices:
        if str(inv.get("status", "")).upper() != "PAID" or inv.pop("_matched", False):
            continue
        amt = float(inv.get("total_amount", 0))
        # 用實際收款日比對；沒有收款日才退回到期日，並放寬容差，
        # 因為付款習性不佳的買方可能晚三個月才匯款，那仍然是正常收款。
        anchor = _d(inv.get("paid_date")) or _d(inv.get("due_date"))
        window = DATE_TOLERANCE_DAYS if inv.get("paid_date") else 120
        found = False
        for k, l in enumerate(inflows):
            if k in used:
                continue
            ldate = _d(l.get("date"))
            if abs(float(l["amount"]) - amt) <= AMOUNT_TOLERANCE and (
                    anchor is None or ldate is None
                    or abs((ldate - anchor).days) <= window):
                used.add(k)
                found = True
                break
        if not found:
            unmatched.append(str(inv.get("invoice_number")))

    paid_n = max(1, sum(1 for i in invoices
                        if str(i.get("status", "")).upper() == "PAID"))
    ratio = 1 - len(unmatched) / paid_n

    # 通過條件是「零筆對不上」，不是「比例夠高」。
    #
    # 原本設 ratio >= 0.8 是錯的：80 張已收款發票裡混一張虛報，
    # 勾稽率仍有 98.7%，檢查照樣通過 —— 造假被平均掉了。
    # 評測時 D15（虛報收款）命中率 0% 就是這樣來的。
    #
    # 在授信場域，「一張對不上」本來就該被指出來讓人查，
    # 而不是因為其他 79 張都對得上就放行。
    # 比例仍然報告出來，因為它反映整體資料品質，但不作為通過條件。
    return [Finding(
        "BANK-01", "收款與銀行流水勾稽", Severity.CRITICAL, not unmatched,
        f"已收款發票中有 {ratio:.0%} 能在銀行流水找到對應入帳"
        f"（其中 {by_ref} 張以發票號碼直接勾稽，屬強證據）。" +
        ("所有已收款發票皆有對應入帳。" if not unmatched else
         f"\n      ⚠ {len(unmatched)} 張標記為已收款、但流水中查無對應入帳，"
         f"每一張都需要說明：" + "、".join(unmatched[:5])
         + ("…" if len(unmatched) > 5 else "")),
        refs=unmatched[:20])]


def check_date_sanity(invoices: list[dict], as_of: Optional[date] = None) -> list[Finding]:
    """
    日期合理性。三種都是「不可能發生」而非「不太可能」，所以是 CRITICAL。
    """
    as_of = as_of or date.today()
    future, reversed_due, too_long = [], [], []
    for i in invoices:
        idate, ddate = _d(i.get("invoice_date")), _d(i.get("due_date"))
        num = str(i.get("invoice_number"))
        if idate and idate > as_of:
            future.append(num)
        if idate and ddate and ddate < idate:
            reversed_due.append(num)
        # 帳期超過 365 天在 B2B 幾乎不存在，多半是打錯或竄改
        if idate and ddate and (ddate - idate).days > 365:
            too_long.append(num)
    return [
        Finding("DATE-01", "發票日期不得為未來", Severity.CRITICAL, not future,
                "無未來日期發票。" if not future else
                f"{len(future)} 張發票的開立日在基準日之後：" + "、".join(future[:5]),
                refs=future[:20]),
        Finding("DATE-02", "到期日不得早於開立日", Severity.CRITICAL, not reversed_due,
                "所有到期日皆晚於開立日。" if not reversed_due else
                f"{len(reversed_due)} 張發票的到期日早於開立日：" + "、".join(reversed_due[:5]),
                refs=reversed_due[:20]),
        Finding("DATE-03", "帳期合理性(≤365天)", Severity.WARNING, not too_long,
                "帳期皆在一年以內。" if not too_long else
                f"{len(too_long)} 張發票帳期超過 365 天，B2B 極罕見，需說明："
                + "、".join(too_long[:5]),
                refs=too_long[:20]),
    ]


def check_amount_sanity(invoices: list[dict]) -> list[Finding]:
    """金額合理性：負數、零、稅額為負。"""
    nonpos, neg_tax, huge = [], [], []
    for i in invoices:
        num = str(i.get("invoice_number"))
        tot = i.get("total_amount")
        tax = i.get("tax_amount")
        if tot is None or float(tot) <= 0:
            nonpos.append(num)
        if tax is not None and float(tax) < 0:
            neg_tax.append(num)
        # 中小企業單筆超過 5000 萬極罕見，值得標記而非直接判錯
        if tot is not None and float(tot) > 50_000_000:
            huge.append(num)
    return [
        Finding("AMT-01", "發票金額須為正數", Severity.CRITICAL, not nonpos,
                "所有發票金額皆為正數。" if not nonpos else
                f"{len(nonpos)} 張發票金額為零或負數：" + "、".join(nonpos[:5]),
                refs=nonpos[:20]),
        Finding("AMT-02", "稅額不得為負", Severity.CRITICAL, not neg_tax,
                "無負稅額。" if not neg_tax else
                f"{len(neg_tax)} 張發票稅額為負：" + "、".join(neg_tax[:5]),
                refs=neg_tax[:20]),
        Finding("AMT-03", "單筆金額量級合理性", Severity.WARNING, not huge,
                "單筆金額量級正常。" if not huge else
                f"{len(huge)} 張發票單筆超過 5,000 萬，中小企業罕見，建議附合約佐證："
                + "、".join(huge[:5]),
                refs=huge[:20]),
    ]


def check_round_numbers(invoices: list[dict]) -> list[Finding]:
    """
    整數金額比例。真實交易的金額因為數量×單價、折扣、運費等，很少剛好是整萬整十萬。
    人為編造的金額則傾向使用整數。這是鑑識會計的常用訊號之一。

    門檻設 25%：真實資料本來就會有一些整數報價（尤其服務業、開口契約），
    設太低會對正常公司誤報。
    """
    if not invoices:
        return []
    amounts = [float(i.get("sales_amount") or 0) for i in invoices]
    amounts = [a for a in amounts if a > 0]
    if not amounts:
        return []
    round_10k = sum(1 for a in amounts if a % 10_000 == 0)
    ratio = round_10k / len(amounts)
    return [Finding(
        "FORENSIC-01", "整數金額比例", Severity.WARNING, ratio < 0.25,
        f"銷售額為萬元整數者占 {ratio:.1%}（{round_10k}/{len(amounts)}）。"
        + ("屬正常範圍。" if ratio < 0.25 else
           "比例偏高。真實交易因數量×單價、折扣、運費，金額很少剛好是整數；"
           "人為編造則傾向使用整數。建議抽查這些發票的出貨單。"))]


def check_benford(invoices: list[dict]) -> list[Finding]:
    """
    班佛定律（Benford's Law）首位數字分布檢定。

    自然產生的財務數據，首位數字為 1 的機率約 30.1%，為 9 的約 4.6%，
    呈對數分布。人為編造的數字則傾向均勻分布。
    這是鑑識會計（forensic accounting）的標準技術，
    美國曾用於偵測選舉舞弊與企業財報造假。

    用卡方檢定量化偏離程度。樣本數少於 50 筆時不做判定 ——
    班佛定律是統計性質，小樣本的偏離沒有意義，硬要判定會製造誤報。
    """
    amounts = [float(i.get("sales_amount") or 0) for i in invoices]
    amounts = [a for a in amounts if a >= 10]          # 個位數金額不適用
    n = len(amounts)
    if n < 50:
        return [Finding(
            "FORENSIC-02", "班佛定律首位數字檢定", Severity.INFO, True,
            f"樣本數 {n} 筆，少於 50 筆的統計門檻，不進行班佛檢定。"
            f"（班佛定律是統計性質，小樣本的偏離沒有意義。）")]

    # ── 適用性前提檢查 ────────────────────────────────────────────────
    # 班佛定律只在資料**跨越足夠多個數量級**時成立。
    # 一家公司若所有發票都在 10 萬～50 萬之間（不到一個數量級），
    # 首位數字本來就不可能呈對數分布，此時做檢定必然誤報。
    #
    # 這個前提是被真實資料驗證出來的：
    #     真實政府決標金額  跨 2.50 decades  χ²=5.14  ✅ 符合班佛
    #     早期合成資料      跨 1.32 decades  χ²=101   ❌ 但那是分布太窄，不是造假
    # 門檻取 1.5 decades（約 30 倍）—— 低於此值，檢定沒有鑑別力。
    span = math.log10(max(amounts) / min(amounts))
    if span < 1.5:
        return [Finding(
            "FORENSIC-02", "班佛定律首位數字檢定", Severity.INFO, True,
            f"金額僅跨越 {span:.2f} 個數量級（{min(amounts):,.0f}~{max(amounts):,.0f}），"
            f"低於 1.5 的適用門檻，不進行班佛檢定。\n"
            f"      班佛定律要求資料跨越多個數量級；範圍過窄時首位數字本來就不會呈對數分布，"
            f"此時做檢定必然誤報。這不是資料有問題，是檢定不適用。")]

    expected_p = [math.log10(1 + 1 / d) for d in range(1, 10)]
    observed = [0] * 9
    for a in amounts:
        first = int(str(int(a))[0])
        if 1 <= first <= 9:
            observed[first - 1] += 1

    chi2 = sum((observed[i] - n * expected_p[i]) ** 2 / (n * expected_p[i])
               for i in range(9))
    # 自由度 8，α=0.05 的臨界值 15.507；α=0.01 為 20.090
    critical_05 = 15.507
    passed = chi2 < critical_05
    dist = "、".join(f"{d+1}:{observed[d]}" for d in range(9))
    return [Finding(
        "FORENSIC-02", "班佛定律首位數字檢定", Severity.WARNING, passed,
        f"樣本 {n} 筆，卡方統計量 χ²={chi2:.2f}（自由度 8，α=0.05 臨界值 15.51）。"
        f"\n      首位數字分布：{dist}"
        + ("\n      分布符合班佛定律，未偵測到人為編造的跡象。" if passed else
           "\n      分布顯著偏離班佛定律。自然產生的財務數字首位為 1 的機率約 30%、"
           "為 9 約 4.6%；人為編造則趨於均勻。建議擴大抽查範圍。"))]


def check_invoice_sequence(invoices: list[dict]) -> list[Finding]:
    """
    同一買方的發票號碼連號偵測。

    真實營運中，開給同一買方的發票會散布在其他買方的發票之間，
    號碼不會連續。大量連號代表這批發票是**同一時間一次補開的** ——
    可能是為了送件而事後補製。
    """
    from collections import defaultdict
    by_buyer: dict[str, list[str]] = defaultdict(list)
    for i in invoices:
        num = str(i.get("invoice_number") or "")
        m = re.search(r"(\d{6,})$", num)
        if m and i.get("buyer_ban"):
            by_buyer[i["buyer_ban"]].append(m.group(1))

    suspicious = []
    for ban, nums in by_buyer.items():
        if len(nums) < 3:
            continue
        vals = sorted(int(x) for x in nums)
        runs = 1
        max_run = 1
        for a, b in zip(vals, vals[1:]):
            runs = runs + 1 if b - a == 1 else 1
            max_run = max(max_run, runs)
        if max_run >= 3:
            suspicious.append(f"{ban}(連號 {max_run} 張)")

    return [Finding(
        "SEQ-01", "同一買方發票連號偵測", Severity.WARNING, not suspicious,
        "未發現同一買方的發票號碼連續。" if not suspicious else
        f"發現 {len(suspicious)} 個買方有連號發票："
        + "、".join(suspicious[:5]) +
        "。真實營運中開給同一買方的發票會散布在其他買方之間；"
        "大量連號代表這批發票可能是同一時間一次補開的。",
        refs=suspicious[:20])]


def check_related_party(invoices: list[dict]) -> list[Finding]:
    """
    關係企業徵兆：買賣方統編前綴高度相似。

    ⚠️ 這是**弱訊號**，刻意設為 INFO 而非 WARNING。
    統編前綴相同不代表是關係企業（統編不是按集團編碼的），
    但在缺乏商工登記負責人資料時，這是唯一能用純程式做的粗篩。
    真正的關係企業偵測需要商工登記的負責人／董監事資料建圖（見 Roadmap）。
    """
    hits = []
    for i in invoices:
        s, b = normalize_tax_id(i.get("seller_ban")), normalize_tax_id(i.get("buyer_ban"))
        if s and b and s != b and s[:4] == b[:4]:
            hits.append(f"{i.get('invoice_number')}({s}/{b})")
    return [Finding(
        "RELATED-01", "關係企業徵兆（統編前綴相似）", Severity.INFO, not hits,
        "未發現買賣方統編前綴高度相似的情形。" if not hits else
        f"{len(hits)} 張發票的買賣方統編前四碼相同：" + "、".join(hits[:5]) +
        "。此為**弱訊號**（統編非按集團編碼），需以商工登記負責人資料進一步查證。",
        refs=hits[:20])]


def check_ledger_integrity(ledger: list[dict]) -> list[Finding]:
    """
    銀行流水的內部一致性：餘額必須等於前一筆餘額加上本筆金額。

    這是最容易被忽略、卻最有效的一項 —— 竄改流水的人通常只改金額，
    忘記把後續所有餘額一起改。
    """
    if len(ledger) < 2:
        return [Finding("LEDGER-01", "銀行流水餘額連續性", Severity.INFO, True,
                        "流水筆數不足，無法檢驗餘額連續性。")]
    broken = []
    prev = None
    for idx, row in enumerate(ledger):
        try:
            bal = float(row.get("balance"))
            amt = float(row.get("amount"))
        except (TypeError, ValueError):
            continue
        if prev is not None and abs((prev + amt) - bal) > AMOUNT_TOLERANCE:
            broken.append(f"第{idx+1}筆({row.get('date')})")
        prev = bal
    return [Finding(
        "LEDGER-01", "銀行流水餘額連續性", Severity.CRITICAL, not broken,
        "所有流水的餘額皆等於前筆餘額加本筆金額。" if not broken else
        f"{len(broken)} 筆流水的餘額不連續：" + "、".join(broken[:5]) +
        "。竄改流水者通常只改金額而忘記重算後續餘額，此為強訊號。",
        refs=broken[:20])]


def check_contract_coverage(invoices: list[dict], contracts: list[dict]) -> list[Finding]:
    """
    重大金額發票是否有合約支撐，以及是否超出年度承諾額。
    """
    if not contracts:
        return [Finding("CONTRACT-01", "重大發票的合約支撐", Severity.INFO, True,
                        "本案未提供合約，無法檢驗合約支撐。銀行受理時通常會要求。")]

    by_buyer = {normalize_tax_id(c.get("buyer_ban")): c for c in contracts if c.get("buyer_ban")}
    # 金額前 25% 視為重大
    amts = sorted((float(i.get("total_amount") or 0) for i in invoices), reverse=True)
    threshold = amts[max(0, len(amts) // 4 - 1)] if amts else 0

    uncovered, over_commit = [], []
    accum: dict[str, float] = {}
    for i in invoices:
        ban = normalize_tax_id(i.get("buyer_ban"))
        amt = float(i.get("total_amount") or 0)
        c = by_buyer.get(ban)
        if amt >= threshold and c is None:
            uncovered.append(str(i.get("invoice_number")))
        if c:
            accum[ban] = accum.get(ban, 0.0) + amt
            commit = float(c.get("annual_commitment_amount") or 0)
            # 24 個月資料對應 2 年承諾額
            if commit and accum[ban] > commit * 2:
                if ban not in [x.split("(")[0] for x in over_commit]:
                    over_commit.append(f"{ban}(累計{accum[ban]:,.0f}/承諾{commit:,.0f}×2)")

    return [
        Finding("CONTRACT-01", "重大發票的合約支撐", Severity.WARNING, not uncovered,
                "重大金額發票皆有對應合約。" if not uncovered else
                f"{len(uncovered)} 張重大金額發票的買方無合約："
                + "、".join(uncovered[:5]) + "。銀行通常要求重大交易附合約。",
                refs=uncovered[:20]),
        Finding("CONTRACT-02", "累計開票 vs 年度承諾額", Severity.WARNING, not over_commit,
                "累計開票未超出合約承諾額。" if not over_commit else
                f"{len(over_commit)} 個買方的累計開票超出合約承諾額："
                + "、".join(over_commit[:3]) + "。可能為虛增營收或合約未更新。",
                refs=over_commit[:20]),
    ]


def check_weekend_issuance(invoices: list[dict]) -> list[Finding]:
    """
    假日開立比例。B2B 交易多在工作日發生；大量假日開票是「一次補製」的訊號。
    門檻設 20%：部分產業（食品、零售供貨）確實有假日出貨，設太嚴會誤報。
    """
    dated = [(_d(i.get("invoice_date")), str(i.get("invoice_number"))) for i in invoices]
    dated = [(d, n) for d, n in dated if d]
    if not dated:
        return []
    weekend = [n for d, n in dated if d.weekday() >= 5]
    ratio = len(weekend) / len(dated)
    return [Finding(
        "FORENSIC-03", "假日開立比例", Severity.WARNING, ratio < 0.20,
        f"假日（週六日）開立者占 {ratio:.1%}（{len(weekend)}/{len(dated)}）。"
        + ("屬正常範圍。" if ratio < 0.20 else
           "比例偏高。B2B 交易多在工作日發生，大量假日開票是「一次補製」的訊號。"),
        refs=weekend[:20])]


def check_concentration(invoices: list[dict]) -> list[Finding]:
    """
    買方集中度。這不是造假偵測，是授信風險評估的標準指標：
    若八成應收都集中在單一買方，該買方一出事企業就跟著倒，
    銀行會據此調降額度或要求加保。誠實揭露比被銀行自己算出來好。
    """
    total = sum(float(i.get("total_amount", 0)) for i in invoices) or 1.0
    by_buyer: dict[str, float] = defaultdict(float)
    for i in invoices:
        by_buyer[i.get("buyer_name") or i.get("buyer_ban") or "未知"] += \
            float(i.get("total_amount", 0))
    top_name, top_amt = max(by_buyer.items(), key=lambda kv: kv[1])
    share = top_amt / total
    return [Finding(
        "RISK-01", "買方集中度", Severity.WARNING, share < 0.5,
        f"最大買方「{top_name}」占應收總額 {share:.1%}（NT${top_amt:,.0f}／"
        f"NT${total:,.0f}）。" + ("集中度在一般可接受範圍。" if share < 0.5 else
        "集中度偏高，銀行通常會要求該買方的信用評等或要求加保，建議事前準備。"))]


CLOSED_STATUSES = {"PAID", "WRITTEN_OFF", "CANCELLED", "VOID"}


def check_overdue(invoices: list[dict], as_of: Optional[date] = None) -> list[Finding]:
    """
    逾期應收與呆帳。兩者刻意分開算，因為在授信上意義完全不同：
      逾期 = 「還沒收到」，是流動性訊號
      呆帳 = 「已認定收不到」，是信用品質訊號
    把已沖銷的呆帳混進逾期率，會讓一家正常公司看起來像要倒了；
    反過來把陳年呆帳一直掛在應收裡不沖銷，則是虛增資產。兩種都會被銀行抓。
    """
    as_of = as_of or date.today()
    overdue, amt = [], 0.0
    written_off, wo_amt = [], 0.0

    for i in invoices:
        status = str(i.get("status", "")).upper()
        if status == "WRITTEN_OFF":
            written_off.append(str(i.get("invoice_number")))
            wo_amt += float(i.get("total_amount", 0))
            continue
        if status in CLOSED_STATUSES:
            continue
        due = _d(i.get("due_date"))
        if due and due < as_of:
            overdue.append(str(i.get("invoice_number")))
            amt += float(i.get("total_amount", 0))

    total_open = sum(float(i.get("total_amount", 0)) for i in invoices
                     if str(i.get("status", "")).upper() not in CLOSED_STATUSES) or 1.0
    billed_total = sum(float(i.get("total_amount", 0)) for i in invoices) or 1.0
    ratio = amt / total_open
    wo_ratio = wo_amt / billed_total

    return [
        Finding("RISK-02", "逾期應收帳款", Severity.WARNING, ratio < 0.30,
                f"逾期未收 {len(overdue)} 筆，合計 NT${amt:,.0f}，占未收帳款 {ratio:.1%}。"
                + ("屬中小企業常見波動範圍。" if ratio < 0.30 else
                   "逾期比偏高，會直接影響信保基金與銀行的徵信評分，"
                   "建議先處理催收或於送件時附上收款計畫。"),
                refs=overdue[:20]),
        Finding("RISK-03", "呆帳沖銷比率", Severity.WARNING, wo_ratio < 0.03,
                f"期間內沖銷呆帳 {len(written_off)} 筆，合計 NT${wo_amt:,.0f}，"
                f"占累計開票金額 {wo_ratio:.2%}。"
                + ("低於一般製造業 3% 的常見水準。" if wo_ratio < 0.03 else
                   "高於一般製造業常見水準，銀行會據此調整風險加碼，"
                   "建議說明個案原因與後續徵信強化措施。"),
                refs=written_off[:20]),
    ]


# ══════════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════════

def run_all(
    invoices: list[dict],
    contracts: Optional[list[dict]] = None,
    ledger: Optional[list[dict]] = None,
    as_of: Optional[date] = None,
) -> dict:
    """
    跑完整套交叉驗證，回傳可直接附在送件文件後面的「證據包」。

    integrity_score 是通過的檢查數加權（critical 權重 3、warning 1、info 0），
    刻意用最簡單、任何人都能自己重算一次的公式 ——
    一個銀行看不懂也算不出來的分數，在授信會議上沒有任何說服力。
    """
    contracts = contracts or []
    ledger = ledger or []

    findings: list[Finding] = []
    # ── 憑證本體（單張就能判定）────────────────────────────────────
    findings += check_tax_ids(invoices)
    findings += check_invoice_arithmetic(invoices)
    findings += check_date_sanity(invoices, as_of)
    findings += check_amount_sanity(invoices)
    # ── 跨憑證（需要看整批才能判定）─────────────────────────────────
    findings += check_self_dealing(invoices)
    findings += check_duplicates(invoices)
    findings += check_invoice_sequence(invoices)
    findings += check_related_party(invoices)
    # ── 跨文件（需要合約或流水才能判定）─────────────────────────────
    findings += check_terms_consistency(invoices, contracts)
    findings += check_contract_coverage(invoices, contracts)
    findings += check_bank_reconciliation(invoices, ledger)
    findings += check_ledger_integrity(ledger)
    # ── 鑑識會計（統計性質，需要足夠樣本）───────────────────────────
    findings += check_round_numbers(invoices)
    findings += check_benford(invoices)
    findings += check_weekend_issuance(invoices)
    # ── 授信風險指標（不是造假偵測，是風險揭露）─────────────────────
    findings += check_concentration(invoices)
    findings += check_overdue(invoices, as_of)

    weight = {Severity.CRITICAL: 3, Severity.WARNING: 1, Severity.INFO: 0}
    earned = sum(weight[f.severity] for f in findings if f.passed)
    possible = sum(weight[f.severity] for f in findings) or 1

    critical_failures = [f for f in findings if not f.passed and f.severity == Severity.CRITICAL]

    return {
        "as_of": str(as_of or date.today()),
        "documents_examined": {
            "invoices": len(invoices), "contracts": len(contracts),
            "bank_transactions": len(ledger),
        },
        "integrity_score": round(earned / possible, 3),
        "critical_failures": len(critical_failures),
        # 有任何 critical 未通過就不該送件 —— 讓結論是一個明確的行動，
        # 而不是一個要使用者自己解讀的分數。
        "submission_ready": len(critical_failures) == 0,
        "findings": [f.as_dict() for f in findings],
    }


def render_text(report: dict) -> str:
    """終端機/報告用的純文字呈現。"""
    lines = ["═" * 78,
             f"  交叉驗證證據包　（基準日 {report['as_of']}）",
             "═" * 78,
             f"  檢視文件：發票 {report['documents_examined']['invoices']} 張、"
             f"合約 {report['documents_examined']['contracts']} 份、"
             f"銀行流水 {report['documents_examined']['bank_transactions']} 筆",
             f"  完整性分數：{report['integrity_score']:.1%}　"
             f"重大缺失：{report['critical_failures']} 項",
             f"  送件建議：{'✅ 可送件' if report['submission_ready'] else '⛔ 建議先補正重大缺失'}",
             "─" * 78]
    icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}
    for f in report["findings"]:
        mark = "✅" if f["passed"] else icon[f["severity"]]
        lines.append(f"  {mark} [{f['check_id']}] {f['title']}")
        lines.append(f"      {f['detail']}")
    lines.append("═" * 78)
    lines.append("  以上每一項皆為程式決定性計算，未經語言模型判斷，可由第三方以相同規則重算驗證。")
    return "\n".join(lines)
