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
    bad: list[str] = []
    for inv in invoices:
        for role in ("seller_ban", "buyer_ban"):
            ban = inv.get(role)
            if ban and not validate_tax_id(ban):
                bad.append(f"{inv.get('invoice_number')}({role}={ban})")
    out.append(Finding(
        "TAXID-01", "統一編號檢核碼", Severity.CRITICAL, not bad,
        "全部統編通過財政部檢核碼演算法。" if not bad else
        f"有 {len(bad)} 筆統編未通過檢核碼，代表號碼本身不可能存在："
        + "、".join(bad[:5]) + ("…" if len(bad) > 5 else ""),
        refs=bad[:20],
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
    return [Finding(
        "BANK-01", "收款與銀行流水勾稽", Severity.CRITICAL, ratio >= 0.8,
        f"已收款發票中有 {ratio:.0%} 能在銀行流水找到對應入帳"
        f"（其中 {by_ref} 張以發票號碼直接勾稽，屬強證據）。" +
        ("" if not unmatched else
         f"另有 {len(unmatched)} 張標記為已收款、但流水中查無對應入帳，需說明："
         + "、".join(unmatched[:5])),
        refs=unmatched[:20])]


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
    findings += check_tax_ids(invoices)
    findings += check_invoice_arithmetic(invoices)
    findings += check_self_dealing(invoices)
    findings += check_duplicates(invoices)
    findings += check_terms_consistency(invoices, contracts)
    findings += check_bank_reconciliation(invoices, ledger)
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
