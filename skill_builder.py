#!/usr/bin/env python3
"""
skill_builder.py — 供應鏈融資領域知識合成器
=============================================================================
把知識庫裡數百份法規、保證要點、銀行商品說明，壓縮成一份給 AI agent 讀的
領域技能檔（SKILL.md）。

【與原版 AnalogGenie 版本的差異，以及為什麼要改】

  1. 領域從「類比 IC 設計」換成「供應鏈融資」，六個合成任務全部重寫。
     核心的 hybrid retrieval + 多樣性過濾引擎沿用，那部分已經驗證過，不重造輪子。

  2. 輸出格式改為 Agent Skills 開放標準（YAML frontmatter + Markdown）。
     這個格式由 Anthropic 於 2025 年提出並開放，目前 Claude Code、Claude.ai、
     OpenAI Codex、Cursor、Gemini CLI 等都支援。
     意思是這份產出不只是「我們系統內部的檔案」，而是可以直接交付給
     合作銀行、讓他們放進自己的 AI 工具裡用的資產。這對 POC 談判很重要 ——
     對方不需要採用我們整套系統，就能先拿到價值。

  3. 加上引用驗證。原版讓 LLM 自己在句尾寫 [Source: xxx]，
     但那串標籤本身也是生成出來的，模型可以在沒讀過那份文件的情況下寫得很像。
     本版會把每段引用拿回檢索文本做字串比對，並在產出的技能檔開頭
     誠實標示驗證通過率 —— 一份宣稱自己「零幻覺」卻沒有驗證數字的知識檔，
     跟一份沒有查核的財報一樣不能用。

  4. 全程繁體中文輸出。原版是英文學術寫作，但這份技能檔的讀者是
     台灣的授信人員與企業主。

用法：
  python skill_builder.py                               # 從 SHARED 知識庫產生
  python skill_builder.py --tenant CASE-0001 -o out/case_skill.md
  python skill_builder.py --model gpt-oss:20b           # 用大模型跑離線深度合成
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flowmind import config, db, evidence, llm, retrieval, textnorm   # noqa: E402
from flowmind.evidence import Claim                                   # noqa: E402

# ══════════════════════════════════════════════════════════════════════════
# System prompt
# ══════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """你是一位在商業銀行企金部門與中小企業信保機構都待過的資深授信主管，
現在負責把散落在法規、保證要點與各行商品說明裡的知識，整理成一份可以交給
初階授信人員與 AI agent 使用的作業知識檔。

【寫作要求】
1. 不要複述文件段落。要做的是跨文件的整合、比較、以及抽取出可操作的判準。
2. 用台灣金融實務的用語：授信、徵信、額度、動撥、帳齡、集中度、債權讓與通知、
   有追索權／無追索權、保證成數、送保、代位清償。不要用中國大陸或翻譯腔用語。
3. 每一個具體數字（成數、費率、天數、金額門檻）都必須附上出處檔名。
   沒有出處的數字寧可不寫 —— 這份檔案會被拿去跟銀行對話。
4. 明確區分「法規強制規定」與「各行慣例」。把慣例寫成規定是實務上很危險的錯誤。
5. 繁體中文。專業、精準、資訊密度高，不要客套話與贅詞。

【輸出格式：嚴格 XML 標籤，供程式解析】
直接以 <section1> 開始，不要有前言。
<section1>
[第一段內容]
</section1>
<section2>
[第二段內容；若該任務標示留白，就輸出空白]
</section2>

【引用格式】
凡引用文件內容，必須以下列格式標記，並且引號內必須是原文逐字複製：
  「原文逐字片段」[來源: 檔名.pdf]

系統會把引號內的字串拿回原文做字串比對，改寫過的引用會被判定為未驗證並公開標示。

引用太長時可以用刪節號「…」省略中間段落，系統支援 ——
但它會把刪節號前後各段**分別**拿去比對，而且要求順序與原文一致。
所以省略沒關係，**改寫不行**；保留下來的每一個字都必須與原文相同。
盡量少用刪節號：片段越多、越短，越容易因為斷在奇怪的位置而驗證失敗。
寧可整段照抄，也不要為了簡潔而重寫。"""


# ══════════════════════════════════════════════════════════════════════════
# 六個合成任務
# ══════════════════════════════════════════════════════════════════════════
def make_tasks() -> dict[str, dict]:
    return {
        "overview": {
            "query": "供應鏈金融 應收帳款承購 中小企業融資 資金缺口 帳期 週轉 "
                     "信用保證 銀行授信 交易真實性",
            "prompt": (
                "<section1>\n"
                "**領域核心命題（200 字內）**\n"
                "先用一句話講清楚：供應鏈金融要解決的根本問題是什麼？\n"
                "格式：\n"
                "**核心命題**：[一句話]\n"
                "**問題結構**：\n"
                "- **誰缺錢**：[中小企業在供應鏈中的位置與現金週期]\n"
                "- **為什麼銀行不願意放**：[資訊不對稱的具體形式]\n"
                "- **供應鏈金融怎麼繞過**：[從『看財報』改成『看交易』的邏輯轉換]\n"
                "- **關鍵前提**：[這個邏輯成立需要什麼條件——交易真實性如何被確認]\n"
                "</section1>\n\n"
                "<section2>\n"
                "**市場結構與各方角色**\n"
                "整理供應鏈金融的參與者及其真正在意的事（不是官方說法，是實際誘因）：\n"
                "核心廠（Anchor）／供應商／銀行／信保機構／平台業者。\n"
                "每一方各寫：他的核心誘因是什麼、他最怕什麼、他願意為什麼付錢。\n"
                "</section2>"
            ),
        },
        "instruments": {
            "query": "應收帳款承購 有追索權 無追索權 供應商融資 信用保證 保證成數 "
                     "手續費 年費率 票貼 額度 動撥 申請文件",
            "prompt": (
                "<section1>\n"
                "**融資工具比較矩陣**\n"
                "輸出一個嚴格的 Markdown 表格，比較各種中小企業可用的短期資金工具。\n"
                "欄位必須是：| 工具 | 擔保／保證基礎 | 典型成數或額度 | 成本結構 | "
                "是否影響資產負債表 | 適用情境 | 主要限制 |\n"
                "至少涵蓋：應收帳款承購（有追索權）、應收帳款承購（無追索權）、"
                "供應商融資（信保基金）、一般營運週轉貸款、票據貼現。\n"
                "有出處的數字才填，沒有的填「未見於本知識庫」，不要用常識推估。\n"
                "</section1>\n\n"
                "<section2>\n"
                "**送件文件檢核清單**\n"
                "整理各工具在受理時實際會被要求的文件，依「必備／視情況／加分」分三級。\n"
                "每一項註明：這份文件是要證明什麼（銀行拿它去驗證哪個風險）。\n"
                "這一節的用途是讓 AI agent 能直接回答企業主的『我還缺什麼』。\n"
                "</section2>"
            ),
        },
        "risk": {
            "query": "徵信 授信審核 買方集中度 帳齡分析 逾期 呆帳 週轉天數 "
                     "交易真實性 虛偽交易 自我交易 風險評估",
            "prompt": (
                "<section1>\n"
                "**授信人員實際在看什麼：判準與紅線**\n"
                "列出 5 到 7 項授信審核時真正的判斷點，每一項寫成可執行的判準：\n"
                "* **判準名稱**：[怎麼算] → [什麼數值算健康／什麼數值會被質疑] → "
                "[這一項想防的風險是什麼]\n"
                "至少涵蓋：買方集中度、帳齡分布、逾期率與呆帳率、"
                "營收與收款的一致性、交易真實性佐證。\n"
                "</section1>\n\n"
                "<section2>\n"
                "**造假樣態與偵測邏輯**\n"
                "整理應收帳款融資領域已知的虛偽交易樣態，每一種寫：\n"
                "手法 → 為什麼有人這樣做 → 從哪些欄位的矛盾可以抓出來 → "
                "純程式可否判定（是／否，若否則說明需要什麼外部資料）。\n"
                "最後一欄很重要：它決定哪些檢查可以自動化、哪些必須人工。\n"
                "</section2>"
            ),
        },
        "regulatory": {
            "query": "中小企業發展條例 中小企業認定標準 債權讓與 民法 營業稅 "
                     "電子發票 商業會計法 個人資料保護法 保存年限",
            "prompt": (
                "<section1>\n"
                "**法規邊界：什麼是強制規定**\n"
                "整理與供應鏈融資直接相關的法規要求，每一條寫：\n"
                "法規名稱與條次 → 規定內容 → 對本業務的實際影響 → 違反的後果。\n"
                "重點涵蓋：中小企業的法定認定標準、債權讓與的成立與對抗要件、"
                "發票與營業稅的開立規定、會計憑證保存年限、個資保護對資料留存的限制。\n"
                "</section1>\n\n"
                "<section2>\n"
                "**可程式化的規則清單**\n"
                "從上述法規中，挑出可以寫成確定性程式檢查的規則（不需要人為判斷的）。\n"
                "格式：規則 → 輸入欄位 → 判定邏輯 → 法源。\n"
                "例如統一編號檢核碼、營業稅率、認定標準的員工數與營業額門檻。\n"
                "這一節會直接對應到 flowmind/crosscheck.py 的實作，是知識轉成程式的介面。\n"
                "</section2>"
            ),
        },
        "gaps": {
            "query": "限制 不適用 除外 排除 不得 爭議 風險 未涵蓋 例外情形",
            "prompt": (
                "<section1>\n"
                "**這個知識庫回答不了的問題**\n"
                "誠實列出 5 項本知識庫涵蓋不足、但實務上會被問到的主題。\n"
                "每一項寫：問題是什麼 → 為什麼現有文件回答不了 → "
                "需要補什麼來源才能回答。\n"
                "不要為了讓知識庫看起來完整而略過這一節 —— "
                "一個不知道自己邊界在哪的 agent，比一個能力較弱但知道界線的更危險。\n"
                "</section1>\n\n"
                "<section2>\n"
                "**Agent 決策樹**\n"
                "給下游 AI agent 的行動指引，格式：\n"
                "- 若【使用者情境】→ 先查【哪類文件】→ 用【什麼判準】→ "
                "但要注意【什麼風險】→ 何時必須轉人工。\n"
                "至少涵蓋：企業主問能借多少、企業主問缺什麼件、"
                "銀行方質疑某筆交易真實性、發現疑似造假、資料不足無法判斷。\n"
                "最後一項特別重要：明確寫出「什麼情況下 agent 應該拒絕回答」。\n"
                "</section2>"
            ),
        },
        "qa": {
            "query": "如何申請 條件 流程 常見問題 差異 比較 什麼情況 需要多久",
            "prompt": (
                "<section1>\n"
                "**實務問答（需跨多份文件整合才答得出來的問題）**\n"
                "回答下列四題，每題都要引用至少兩個不同來源：\n"
                "**Q1：一家帳期 90 天、最大買方占營收 60% 的精密機械廠，"
                "適合走應收帳款承購還是信保基金的供應商融資？兩者的取捨在哪？**\n"
                "**Q2：無追索權承購真的能把應收帳款移出資產負債表嗎？"
                "在什麼條件下不行？**\n"
                "**Q3：銀行說『這筆交易真實性存疑』時，企業實際上要拿出什麼來反駁？"
                "哪些文件的證明力最強？**\n"
                "**Q4：一家剛成立兩年、沒有完整財報的公司，"
                "在現行制度下有哪些路可以走？**\n"
                "每題答案結尾標註：這個答案的最大不確定性在哪。\n"
                "</section1>\n\n"
                "<section2>\n"
                "留白。\n"
                "</section2>"
            ),
        },
    }


# ══════════════════════════════════════════════════════════════════════════
# 引用抽取與驗證
# ══════════════════════════════════════════════════════════════════════════
_SOURCE_TAG = re.compile(r"\[來源[:：]\s*([^\]]+)\]")
_QUOTE = re.compile(r"「([^「」]{6,400})」")
# 引文與來源標籤之間允許隔多少字元。模型常寫成
#   「…」<br><u>[來源: x]</u>　或　「…」（法源依據：[來源: x]）
# 這類 markdown/HTML 裝飾，甚至把兩者放進表格的不同欄位。
LOOKBACK_CHARS = 260


def extract_and_verify(text: str, chunks: list[retrieval.Chunk]) -> tuple[list[Claim], float]:
    """
    把生成內容裡的「原文引用」與 [來源: 檔名] 配對，回原文做字串比對。

    配對方式是「從來源標籤往回找最近的一段引號引文」，而不是要求兩者字面相鄰。
    這一版改過：原本要求 `「…」[來源: x]` 緊鄰，實測 84 段引文裡只抓到 6 段 ——
    模型會在中間夾 `<br><u>` 之類的 markdown/HTML 裝飾，或放進表格的不同欄位。
    抓不到就等於沒驗證，而「沒驗證」在報告上會顯示成「驗證通過率 0%」，
    看起來像模型在亂寫，其實是抽取器太死。

    要強調的是：**這只放寬了「哪句話宣稱出自哪份文件」的判讀，
    沒有放寬驗證本身。** 每一段配對出來的引文，仍然要逐字回原文比對才算通過。
    """
    claims: list[Claim] = []
    seen: set[tuple[str, str]] = set()

    for tag in _SOURCE_TAG.finditer(text):
        source = tag.group(1).strip()
        window = text[max(0, tag.start() - LOOKBACK_CHARS):tag.start()]
        quotes = _QUOTE.findall(window)
        if not quotes:
            continue
        quote = quotes[-1].strip()          # 最靠近標籤的那一段
        key = (quote[:80], source)
        if key in seen:
            continue
        seen.add(key)
        claims.append(Claim(statement="", quote=quote, source=source))

    if not claims:
        return [], 0.0
    evidence.verify_claims(claims, chunks)
    return claims, evidence.citation_integrity(claims)


def extract_xml(text: str, tag: str) -> str:
    cleaned = re.sub(r"^```(?:xml|markdown)?\s*$", "", text, flags=re.MULTILINE)
    m = re.search(f"<{tag}>(.*?)</{tag}>", cleaned, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    if tag == "section1" and len(cleaned.strip()) > 80:
        return (f"> *[系統註記：模型未輸出 XML 標籤，以下為原始輸出]*\n\n"
                f"{cleaned.strip()}")
    return ""


# ══════════════════════════════════════════════════════════════════════════
# 任務執行
# ══════════════════════════════════════════════════════════════════════════
def run_task(name: str, task: dict, tenant_id: str, model: str) -> dict:
    t0 = time.time()
    with db.tenant_session(tenant_id) as conn:
        chunks = retrieval.hybrid_search(conn, task["query"], top_k=16, max_per_source=4)

    if not chunks:
        return {"name": name, "s1": "[知識庫中無相關文件]", "s2": "",
                "sources": [], "claims": [], "integrity": 0.0, "elapsed": 0.0}

    context = "\n\n---\n\n".join(
        f"[來源: {c.source} | 類別: {c.category} | RRF {c.rrf_score:.4f}]\n{c.parent_content}"
        for c in chunks)

    try:
        # 走 chat_local 而不是 chat：需要設定 num_ctx，否則長 context 會被
        # Ollama 靜默截斷（見 flowmind/llm.py 的 chat_local 說明）
        raw = llm.chat_local(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": f"【檢索文本】\n{context}\n\n{task['prompt']}"}],
            model=model, role="synth", temperature=0.2, num_ctx=32768)
    except Exception as e:                            # noqa: BLE001
        print(f"   ❌ [{name}] 生成失敗：{e}")
        return {"name": name, "s1": f"[生成失敗：{e}]", "s2": "",
                "sources": sorted({c.source for c in chunks}),
                "claims": [], "integrity": 0.0, "elapsed": time.time() - t0}

    # 簡轉繁要在引用抽取「之前」做。
    # 我們的來源文件全部是繁體中文的台灣公文與行庫資料，
    # 把模型漏出的簡體字轉回繁體，只會讓引用比對更準，不會更寬鬆。
    raw = textnorm.to_traditional(raw)

    s1, s2 = extract_xml(raw, "section1"), extract_xml(raw, "section2")
    if not s1.strip():
        # 不要讓「模型輸出了東西但我們解析不到」跟「模型什麼都沒輸出」長得一樣。
        # 這兩種失敗的處理方式完全不同，混在一起會查很久。
        print(f"   ⚠️  [{name}] 模型回了 {len(raw)} 字元但抽不到 <section1>，"
              f"開頭：{raw[:120]!r}")
    claims, integrity = extract_and_verify(s1 + "\n" + s2, chunks)

    elapsed = time.time() - t0
    print(f"   ✅ [{name}] {elapsed:.0f}s｜來源 {len({c.source for c in chunks})} 份｜"
          f"引用 {len(claims)} 條，驗證通過 {integrity:.0%}")
    return {"name": name, "s1": s1, "s2": s2, "elapsed": elapsed,
            "sources": sorted({c.source for c in chunks}),
            "claims": claims, "integrity": integrity,
            "peak_rrf": {c.source: c.rrf_score for c in chunks}}


# ══════════════════════════════════════════════════════════════════════════
# 產出 SKILL.md
# ══════════════════════════════════════════════════════════════════════════
def build(tenant_id: str, output: Path, model: str, workers: int) -> None:
    with db.tenant_session(tenant_id) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT source, metadata->>'category', COUNT(*) "
                        "FROM documents GROUP BY 1,2 ORDER BY 1")
            inventory = cur.fetchall()
    if not inventory:
        print(f"❌ engagement={tenant_id} 的知識庫是空的。"
              f"請先執行：python data_update_finance.py --tenant {tenant_id} --rebuild")
        return

    all_sources = {row[0]: row[1] or "未分類" for row in inventory}
    print(f"\n{'═'*72}\n  🧠 供應鏈融資領域知識合成"
          f"\n  知識庫 {len(all_sources)} 份文件｜模型 {model}｜engagement {tenant_id}\n{'═'*72}\n")

    t_total = time.time()
    tasks = make_tasks()
    results: dict[str, dict] = {}

    # 平行度刻意設低（預設 2）：本地 8GB VRAM 同時跑六條會讓 Ollama
    # 反覆換入換出模型，總時間反而更長。這不是保守，是實測後的取捨。
    print(f"   ⚡ 啟動 {workers} 條合成工作緒（本地 GPU 顯存有限，刻意不開滿）\n")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(run_task, n, t, tenant_id, model): n for n, t in tasks.items()}
        for f in concurrent.futures.as_completed(futures):
            n = futures[f]
            try:
                results[n] = f.result()
            except Exception as e:                    # noqa: BLE001
                print(f"   ❌ [{n}] 失敗：{e}")
                results[n] = {"name": n, "s1": f"[合成失敗：{e}]", "s2": "",
                              "sources": [], "claims": [], "integrity": 0.0}

    # ── 覆蓋率與引用驗證統計 ──────────────────────────────────────────
    hit_sources = {s for r in results.values() for s in r.get("sources", [])}
    missed = sorted(set(all_sources) - hit_sources)
    all_claims = [c for r in results.values() for c in r.get("claims", [])]
    verified = sum(1 for c in all_claims if c.is_grounded)
    integrity = verified / len(all_claims) if all_claims else 0.0
    elapsed = time.time() - t_total

    def sec(name: str, key: str = "s1") -> str:
        return results.get(name, {}).get(key) or "*（本節未產生內容）*"

    md = f"""---
name: taiwan-supply-chain-finance
description: >-
  台灣中小企業供應鏈融資的作業知識：應收帳款承購（有／無追索權）、
  信保基金供應商融資、授信徵信判準、造假樣態偵測、以及相關法規邊界。
  當使用者詢問中小企業如何取得短期營運資金、銀行授信會看什麼、
  應收帳款可否融資、送件需要哪些文件時使用本技能。
  所有具體數字均附出處，並標示引用驗證通過率。
license: 內部使用；引用之公開資料著作權歸各原始機關所有
---

# 台灣中小企業供應鏈融資：作業知識

> ⚠️ **使用前必讀**
> 本技能提供的是**盡職調查與資訊整理**的支援，不構成授信決策、
> 不構成投資或財務建議。任何對外提出的融資建議與額度判斷，
> 必須由具備授信權責的人員覆核並簽署。
> 本檔案由 `skill_builder.py` 從知識庫自動合成，內容正確性受限於下方揭露的來源清單。

## 產出資訊（透明度揭露）

| 項目 | 數值 |
|---|---|
| 知識來源 | {len(all_sources)} 份文件（{len(hit_sources)} 份被實際檢索到，覆蓋率 {len(hit_sources)/len(all_sources):.0%}） |
| 引用條數 | {len(all_claims)} 條 |
| **引用驗證通過率** | **{integrity:.1%}**（每條引用皆回原文做字串比對，非模型自評） |
| 檢索方式 | Hybrid RRF：稠密向量 `bge-m3` + 中文 bigram BM25，多樣性上限 4 chunk/來源 |
| 合成模型 | `{model}`（本地 Ollama，資料不出本機） |
| Engagement | `{tenant_id}` |
| 產生日期 | {date.today()} |

> 引用驗證通過率是這份檔案最重要的一個數字。
> 它代表：文中打上引號並標註出處的句子裡，有多少比例真的能在原始文件中逐字找到。
> 沒有這個數字的知識檔，等同於沒有查核的財報。

---

## 一、領域核心命題

{sec('overview')}

### 市場結構與各方誘因

{sec('overview', 's2')}

---

## 二、融資工具比較

{sec('instruments')}

### 送件文件檢核清單

{sec('instruments', 's2')}

---

## 三、授信判準與風險評估

{sec('risk')}

### 造假樣態與偵測邏輯

{sec('risk', 's2')}

---

## 四、法規邊界

{sec('regulatory')}

### 可程式化的規則清單

{sec('regulatory', 's2')}

> 本節列出的規則已有部分實作於 `flowmind/crosscheck.py`，
> 以純 Python 決定性運算完成，不經語言模型判斷。

---

## 五、實務問答

{sec('qa')}

---

## 六、知識邊界（本技能回答不了的問題）

{sec('gaps')}

### Agent 決策樹

{sec('gaps', 's2')}

---

## 未被檢索到的來源（{len(missed)} 份）

以下文件存在於知識庫，但在本次合成的六組查詢中未被檢索命中，
代表上述內容**不包含**這些文件的資訊：

{chr(10).join(f"- `{s}`（{all_sources[s]}）" for s in missed) or "（全部來源皆被檢索到）"}

> 給下游 agent 的指示：當問題涉及上列文件的主題時，
> 你的信心應該降低，並主動觸發一次針對性檢索來補足，而不是依賴本檔案作答。

---

## 完整來源清單

{chr(10).join(f"- `{s}` — {all_sources[s]}" for s in sorted(all_sources))}

---

*由 `skill_builder.py` 自動合成，耗時 {elapsed:.0f} 秒。*
*格式遵循 Agent Skills 開放標準，可直接置於 `.claude/skills/` 下使用。*
"""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(md, encoding="utf-8")

    # 交付前自我檢查：確認沒有簡體字漏網。
    # 這份檔案可能被交給合作銀行，出現簡體字會讓人懷疑資料來源的正當性。
    leftover = textnorm.count_simplified(md)
    if leftover:
        print(f"  ⚠️  仍偵測到簡體字 {leftover}，請人工檢查後再對外交付。")

    # 引用驗證明細另存，供人工抽查
    detail = [{"quote": c.quote[:200], "source": c.source,
               "verdict": c.verdict.value, "score": c.match_score,
               "matched_source": c.matched_source} for c in all_claims]
    (output.parent / (output.stem + "_citations.json")).write_text(
        json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'═'*72}")
    print(f"  ✅ SKILL.md → {output}")
    print(f"  📊 來源覆蓋 {len(hit_sources)}/{len(all_sources)}"
          f"｜引用 {len(all_claims)} 條，驗證通過 {integrity:.1%}｜{elapsed:.0f}s")
    if integrity < 0.7 and all_claims:
        print(f"  ⚠️  引用驗證通過率偏低。這通常代表模型在改寫原文而非逐字引用，")
        print(f"      或知識庫缺乏支撐這些主張的文件。發布前建議人工抽查 "
              f"{output.stem}_citations.json")
    print(f"{'═'*72}\n")


def main():
    ap = argparse.ArgumentParser(description="供應鏈融資領域知識合成器")
    ap.add_argument("--tenant", "-t", default="SHARED",
                    help="從哪個 engagement 的知識庫合成（預設 SHARED 公開知識庫）")
    ap.add_argument("--output", "-o", default="out/skills/taiwan-supply-chain-finance/SKILL.md")
    ap.add_argument("--model", "-m", default=None, help=f"預設 {config.SYNTH_MODEL}")
    ap.add_argument("--workers", "-w", type=int, default=2)
    args = ap.parse_args()
    build(args.tenant, Path(args.output), args.model or config.SYNTH_MODEL, args.workers)


if __name__ == "__main__":
    main()
