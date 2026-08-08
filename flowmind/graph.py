"""
flowmind.graph — 知識圖譜建構、多跳查詢與 Obsidian 匯出
=============================================================================
【這一層解的是向量檢索在原理上解不了的問題】

見 docs/DECISIONS.md U-01：
「日本的保證成數」與「台灣的保證成數」在語意空間裡幾乎重合，
embedding 分不出指涉對象是誰。範圍詞驗證（字串比對）只能攔 3/8。

真正的解法是把「指涉對象」變成**結構**：
文件 -[applies_to]-> 法域(台灣) / 期間(2025) / 主題(供應商融資)

一旦是結構，「日本 ≠ 台灣」就是一次 JOIN，不是相似度的事。

【圖上有什麼】

  法域 jurisdiction   台灣 / 日本 / 美國 …
  期間 period         2025 / 115年07月 …
  主題 topic          供應商融資 / 應收帳款承購 / 營業稅 …
  文件 document       每份知識庫文件
  企業 company        依統一編號建立（真實可查證的識別碼）
  自然人 person       公司負責人（來源：經濟部商工登記公示資料）
  發票 invoice        每張憑證

【邊】

  document -applies_to-> jurisdiction / period / topic
  document -supersedes-> document          （版本關係）
  company  -issued_to->  company           （開票，weight = 金額）
  company  -represented_by-> person        （負責人）
  invoice  -party_to->   company

【設計原則：圖是索引，不是事實來源】
節點與邊都從既有資料推導，不手動維護 —— 手動維護的圖三個月後一定跟資料脫節。
`rebuild()` 是冪等的：每次執行後圖完整反映當下的資料狀態。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import psycopg2.extras

from . import config, db

# ══════════════════════════════════════════════════════════════════════════
# 適用範圍 vs 提及：這個區分是整個模組存在的理由
# ══════════════════════════════════════════════════════════════════════════
# 第一版用「文中提到某國幾次」來推導 applies_to，結果在它專門要解決的案例上
# 直接失敗：中小企業白皮書提到「日本」25~35 次（當作國際比較），
# 於是被判定為「適用於日本」——問「日本的保證成數」時系統說可以回答。
#
# **用出現頻率推導適用範圍，只是字串比對的華麗版本，failure mode 完全一樣。**
#
# 真正可靠的訊號是**結構性的**：文件的發布機關決定它的管轄範圍。
#   全國法規資料庫 / 信保基金 / 財政部 / 經濟部 / 台灣的銀行 → 適用於台灣
#   白皮書提到日本 → 那是**提及（mentions）**，不是適用範圍
#
# 所以拆成兩種邊：
#   applies_to  適用於 —— 由發布機關決定，是「這份文件管的是哪裡」
#   mentions    提及   —— 由文字頻率決定，是「這份文件談到了哪裡」
#
# 回答「日本的保證成數」時只看 applies_to：知識庫沒有任何一份文件適用於日本，
# 所以拒答。同時可以用 mentions 告訴使用者「有 3 份文件提到日本，但那是比較性
# 敘述，不是日本的制度」—— 這比單純拒答有用得多。
JURISDICTIONS = {
    "台灣": ["中華民國", "臺灣", "台灣", "本國", "國內"],
    "日本": ["日本"], "韓國": ["韓國", "南韓"], "新加坡": ["新加坡"],
    "香港": ["香港"], "中國大陸": ["中國大陸", "大陸地區"],
    "美國": ["美國"], "歐盟": ["歐盟", "歐洲聯盟"], "英國": ["英國"],
}

# 發布機關 → 管轄法域。這是 applies_to 的唯一來源。
# 機關本身就決定了管轄範圍，這是制度事實，不需要從文字推論。
PUBLISHER_JURISDICTION = {
    "全國法規資料庫": "台灣",
    "中小企業信用保證基金": "台灣",
    "財政部": "台灣",
    "經濟部中小及新創企業署": "台灣",
    "中國信託商業銀行": "台灣",
    "玉山商業銀行": "台灣",
    "永豐商業銀行": "台灣",
    "Taiwan SMEG": "台灣",
    "U.S. Small Business Administration": "美國",
}

# ── 主題偵測 ─────────────────────────────────────────────────────────
TOPICS = {
    "供應商融資": ["供應商融資"],
    "應收帳款承購": ["應收帳款承購", "帳款承購", "factoring"],
    "信用保證": ["信用保證", "保證成數", "送保"],
    "營業稅": ["營業稅", "統一發票"],
    "債權讓與": ["債權讓與"],
    "會計憑證": ["會計憑證", "商業會計"],
    "個人資料保護": ["個人資料", "個資"],
    "中小企業認定": ["認定標準", "中小企業之認定"],
    "承保統計": ["承保統計", "承保融資金額"],
}

_ROC_YEAR = re.compile(r"(?:民國\s*)?(\d{2,3})\s*年")
_AD_YEAR = re.compile(r"(20\d{2})\s*年?")


def _nid(kind: str, key: str) -> str:
    """節點 ID：型別 + 內容雜湊。同一個實體在不同來源會得到同一個 ID。"""
    h = hashlib.sha1(f"{kind}|{key}".encode("utf-8")).hexdigest()[:16]
    return f"{kind}:{h}"


@dataclass
class GraphStats:
    nodes: dict[str, int]
    edges: dict[str, int]

    @property
    def total_nodes(self) -> int:
        return sum(self.nodes.values())

    @property
    def total_edges(self) -> int:
        return sum(self.edges.values())


# ══════════════════════════════════════════════════════════════════════════
# 建構
# ══════════════════════════════════════════════════════════════════════════

def _upsert_nodes(cur, rows: list[tuple]) -> None:
    if rows:
        psycopg2.extras.execute_values(cur, """
            INSERT INTO kg_nodes (node_id, node_type, label, tenant_id, attrs)
            VALUES %s ON CONFLICT (node_id) DO UPDATE
              SET label = EXCLUDED.label, attrs = EXCLUDED.attrs
        """, rows, template="(%s,%s,%s,%s,%s::jsonb)")


def _upsert_edges(cur, rows: list[tuple]) -> None:
    """
    寫入邊。同一批次內的重複邊必須先在 Python 端聚合 ——
    PostgreSQL 的 ON CONFLICT DO UPDATE 不允許同一個命令內
    對同一列更新兩次（CardinalityViolation）。

    而重複是常態：同一個買方本來就會有很多張發票，
    它們全部聚合成一條「開票給」的邊，weight 是累計金額。
    這正是圖相對於逐筆紀錄的價值 —— 看的是關係強度，不是單筆交易。
    """
    if not rows:
        return
    agg: dict[tuple, list] = {}
    for src, dst, etype, tenant, weight, attrs in rows:
        key = (src, dst, etype, tenant)
        if key in agg:
            agg[key][0] += float(weight or 0)
            agg[key][1] += 1
        else:
            agg[key] = [float(weight or 0), 1, attrs]

    merged = [(k[0], k[1], k[2], k[3], v[0],
               json.dumps({"count": v[1]}, ensure_ascii=False))
              for k, v in agg.items()]
    psycopg2.extras.execute_values(cur, """
        INSERT INTO kg_edges (src_id, dst_id, edge_type, tenant_id, weight, attrs)
        VALUES %s ON CONFLICT (src_id, dst_id, edge_type, tenant_id) DO UPDATE
          SET weight = EXCLUDED.weight, attrs = EXCLUDED.attrs
    """, merged, template="(%s,%s,%s,%s,%s,%s::jsonb)")


def build_shared_graph() -> GraphStats:
    """
    從 SHARED 知識庫建立「文件 → 法域／期間／主題」的範圍圖。

    這是 U-01 的解法：把每份文件講的是「哪個法域、哪個年度、哪個主題」
    抽成結構。之後問「日本的保證成數」，就能用 JOIN 判斷
    知識庫裡有沒有 applies_to 日本的文件 —— 而不是靠相似度猜。
    """
    nodes: dict[str, tuple] = {}
    edges: list[tuple] = []

    # 發布機關來自資料來源登錄表（人工確認過的），不是從文件內容猜的。
    # 這是 applies_to 唯一可靠的依據。
    reg_path = config.DATA_DIR / "sources_registry.json"
    registry: dict[str, dict] = {}
    if reg_path.exists():
        registry = {e["filename"]: e
                    for e in json.loads(reg_path.read_text(encoding="utf-8"))["entries"]}

    with db.tenant_session("SHARED") as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT source,
                       string_agg(content, ' ') AS body,
                       (array_agg(metadata))[1] AS meta
                FROM documents WHERE tenant_id='SHARED'
                GROUP BY source
            """)
            docs = cur.fetchall()

        for source, body, meta in docs:
            meta = meta if isinstance(meta, dict) else {}
            doc_id = _nid("document", source)
            nodes[doc_id] = (doc_id, "document", source, "SHARED", json.dumps({
                "category": meta.get("category"),
                "published": meta.get("published"),
                "doc_status": meta.get("doc_status"),
                "authority": meta.get("authority"),
            }, ensure_ascii=False))

            text = f"{source} {body[:120000]}"

            # ── applies_to：由**發布機關**決定，不從文字推論 ──────────
            publisher = registry.get(source, {}).get("publisher")
            juris_of_doc = PUBLISHER_JURISDICTION.get(publisher or "")
            if juris_of_doc:
                jid = _nid("jurisdiction", juris_of_doc)
                nodes[jid] = (jid, "jurisdiction", juris_of_doc, "SHARED", "{}")
                edges.append((doc_id, jid, "applies_to", "SHARED", 1.0,
                              json.dumps({"basis": f"發布機關：{publisher}"},
                                         ensure_ascii=False)))

            # ── mentions：由文字頻率決定，**與適用範圍嚴格分開** ────────
            # 白皮書提到日本 25 次是「提及」，不代表它講的是日本的制度。
            for juris, kws in JURISDICTIONS.items():
                if juris == juris_of_doc:
                    continue                       # 本國不重複記為提及
                hits = sum(text.count(k) for k in kws)
                if hits < 3:
                    continue
                jid = _nid("jurisdiction", juris)
                nodes[jid] = (jid, "jurisdiction", juris, "SHARED", "{}")
                edges.append((doc_id, jid, "mentions", "SHARED", float(hits), "{}"))

            # 主題
            for topic, kws in TOPICS.items():
                hits = sum(text.count(k) for k in kws)
                if hits < 2:
                    continue
                tid = _nid("topic", topic)
                nodes[tid] = (tid, "topic", topic, "SHARED", "{}")
                edges.append((doc_id, tid, "applies_to", "SHARED", float(hits), "{}"))

            # 期間：優先用登錄表的發布時間，其次從檔名推
            pub = meta.get("published")
            if pub:
                pid = _nid("period", str(pub))
                nodes[pid] = (pid, "period", str(pub), "SHARED", "{}")
                edges.append((doc_id, pid, "applies_to", "SHARED", 1.0, "{}"))

            # 版本取代關係
            sup = meta.get("superseded_by")
            if sup:
                nid2 = _nid("document", sup)
                nodes.setdefault(nid2, (nid2, "document", sup, "SHARED", "{}"))
                edges.append((nid2, doc_id, "supersedes", "SHARED", 1.0, "{}"))

        with conn.cursor() as cur:
            _upsert_nodes(cur, list(nodes.values()))
            _upsert_edges(cur, edges)
        conn.commit()

    return _stats("SHARED")


def build_engagement_graph(tenant_id: str) -> GraphStats:
    """
    從委任案的憑證建立交易圖：企業節點 + 開票邊（weight = 累計金額）。

    有了這張圖，循環交易（A 開票給 B，B 也開票給 A）就變成一次自連接查詢 ——
    而那是純憑證比對抓不到的：兩張發票各自完全合法。
    """
    base = config.RAW_DIR / tenant_id
    if not (base / "receivables.json").exists():
        return GraphStats({}, {})

    invoices = json.loads((base / "receivables.json").read_text(encoding="utf-8"))
    payables_p = base / "payables.json"
    payables = json.loads(payables_p.read_text(encoding="utf-8")) if payables_p.exists() else []

    nodes: dict[str, tuple] = {}
    edges: list[tuple] = []

    def company(ban: Optional[str], name: Optional[str]) -> Optional[str]:
        if not ban:
            return None
        cid = _nid("company", str(ban))
        nodes[cid] = (cid, "company", name or str(ban), tenant_id,
                      json.dumps({"ban": ban}, ensure_ascii=False))
        return cid

    for inv in invoices:
        s = company(inv.get("seller_ban"), inv.get("seller_name"))
        b = company(inv.get("buyer_ban"), inv.get("buyer_name"))
        if s and b:
            edges.append((s, b, "issued_to", tenant_id,
                          float(inv.get("total_amount") or 0),
                          json.dumps({"doc": inv.get("invoice_number")},
                                     ensure_ascii=False)))

    # 應付帳款：我方 → 供應商，方向相反，讓循環交易偵測能看到雙向
    for p in payables:
        b = company(p.get("buyer_ban"), p.get("buyer_name"))
        s = company(p.get("supplier_ban"), p.get("supplier_name"))
        if s and b:
            edges.append((s, b, "issued_to", tenant_id,
                          float(p.get("amount") or 0),
                          json.dumps({"doc": p.get("bill_number")},
                                     ensure_ascii=False)))

    with db.tenant_session(tenant_id) as conn:
        with conn.cursor() as cur:
            _upsert_nodes(cur, list(nodes.values()))
            _upsert_edges(cur, edges)
        conn.commit()
    return _stats(tenant_id)


def add_representative(tenant_id: str, company_ban: str, company_name: str,
                       person_name: str) -> None:
    """
    加入公司負責人關係。資料來源應為經濟部商工登記公示資料。

    這是關係企業偵測的**正解** —— 目前 crosscheck 的 RELATED-01
    只能靠統編前綴比對（弱訊號，統編根本不是按集團編碼的）。
    有了負責人資料，「A 與 B 的負責人是同一人」是確定性的事實。
    """
    cid, pid = _nid("company", company_ban), _nid("person", person_name)
    with db.tenant_session(tenant_id) as conn:
        with conn.cursor() as cur:
            _upsert_nodes(cur, [
                (cid, "company", company_name, tenant_id,
                 json.dumps({"ban": company_ban}, ensure_ascii=False)),
                (pid, "person", person_name, tenant_id, "{}")])
            _upsert_edges(cur, [(cid, pid, "represented_by", tenant_id, 1.0,
                                 json.dumps({"source": "商工登記公示資料"},
                                            ensure_ascii=False))])
        conn.commit()


def _stats(tenant_id: str) -> GraphStats:
    with db.tenant_session(tenant_id) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT node_type, COUNT(*) FROM kg_nodes GROUP BY 1")
            n = dict(cur.fetchall())
            cur.execute("SELECT edge_type, COUNT(*) FROM kg_edges GROUP BY 1")
            e = dict(cur.fetchall())
    return GraphStats(n, e)


# ══════════════════════════════════════════════════════════════════════════
# 查詢
# ══════════════════════════════════════════════════════════════════════════

def scope_check(question: str, jurisdictions: list[str],
                topics: list[str]) -> dict:
    """
    ★ U-01 的解法：問題指涉的法域／主題，知識庫裡到底有沒有對應文件？

    這與 evidence.missing_scope_terms（字串比對）的差別在於：
    字串比對只能回答「文本裡有沒有出現『日本』」，
    圖能回答「有沒有一份**適用於日本**的文件」—— 這才是問題真正在問的。

    白皮書提到日本當作比較對象，字串比對會誤判為「有」；
    圖上那份文件的 applies_to 邊指向台灣而非日本，所以不會誤判。
    """
    out: dict[str, Any] = {"question": question, "jurisdictions": {}, "topics": {}}
    with db.tenant_session("SHARED") as conn:
        with conn.cursor() as cur:
            for j in jurisdictions:
                # 只看 applies_to（發布機關決定的管轄範圍）
                cur.execute("""
                    SELECT d.label
                    FROM kg_edges e
                    JOIN kg_nodes j ON j.node_id = e.dst_id AND j.node_type='jurisdiction'
                    JOIN kg_nodes d ON d.node_id = e.src_id AND d.node_type='document'
                    WHERE e.edge_type='applies_to' AND j.label = %s
                    LIMIT 5
                """, (j,))
                applies = [r[0] for r in cur.fetchall()]
                # mentions 另外查：用來對使用者說明「有文件提到但那不是它的制度」
                cur.execute("""
                    SELECT d.label, e.weight
                    FROM kg_edges e
                    JOIN kg_nodes j ON j.node_id = e.dst_id AND j.node_type='jurisdiction'
                    JOIN kg_nodes d ON d.node_id = e.src_id AND d.node_type='document'
                    WHERE e.edge_type='mentions' AND j.label = %s
                    ORDER BY e.weight DESC LIMIT 5
                """, (j,))
                mentions = [{"source": s, "mentions": float(w)} for s, w in cur.fetchall()]
                out["jurisdictions"][j] = {
                    "covered": bool(applies),
                    "documents": [{"source": s} for s in applies],
                    "mentioned_in": mentions,
                }
            for t in topics:
                cur.execute("""
                    SELECT d.label, e.weight
                    FROM kg_edges e
                    JOIN kg_nodes tp ON tp.node_id = e.dst_id AND tp.node_type='topic'
                    JOIN kg_nodes d  ON d.node_id  = e.src_id AND d.node_type='document'
                    WHERE e.edge_type='applies_to' AND tp.label = %s
                    ORDER BY e.weight DESC LIMIT 5
                """, (t,))
                docs = cur.fetchall()
                out["topics"][t] = {
                    "covered": bool(docs),
                    "documents": [{"source": s, "mentions": float(w)} for s, w in docs],
                }
    uncovered = [j for j, v in out["jurisdictions"].items() if not v["covered"]]
    out["uncovered_jurisdictions"] = uncovered
    out["answerable"] = not uncovered
    return out


def multi_hop(tenant_id: str, start_label: str, max_hops: int = 3,
              edge_types: Optional[list[str]] = None) -> list[dict]:
    """從某個節點出發做多跳查詢。回傳含路徑，讓結果可以被追溯。"""
    with db.tenant_session(tenant_id) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT node_id, label FROM kg_nodes WHERE label = %s LIMIT 1",
                        (start_label,))
            row = cur.fetchone()
            if not row:
                return []
            cur.execute("SELECT * FROM kg_neighbors(%s, %s, %s)",
                        (row["node_id"], max_hops, edge_types))
            return [dict(r) for r in cur.fetchall()]


def circular_trades(tenant_id: str) -> list[dict]:
    with db.tenant_session(tenant_id) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM kg_find_circular_trades()")
            return [dict(r) for r in cur.fetchall()]


def shared_representatives(tenant_id: str) -> list[dict]:
    with db.tenant_session(tenant_id) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM kg_find_shared_representatives()")
            return [dict(r) for r in cur.fetchall()]


# ══════════════════════════════════════════════════════════════════════════
# Obsidian 匯出：Map of Content
# ══════════════════════════════════════════════════════════════════════════

_FS_UNSAFE = re.compile(r'[\\/:*?"<>|]')


def export_obsidian(tenant_id: str, out_dir: Path) -> dict:
    """
    把圖匯出成 Obsidian vault：每個節點一個 note，用 [[wikilink]] 連接。

    為什麼值得做：Obsidian 的關係圖是**人可以用滑鼠探索**的介面。
    一份 JSON 或一張 SQL 查詢結果，授信人員不會去看；
    一張可以點開、可以搜尋、可以看到「這家公司牽連到誰」的圖，他會。

    每個 note 的 frontmatter 帶屬性（統編、類別、發布時間、狀態），
    Obsidian 的 Dataview 外掛可以直接對這些屬性下查詢。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {"nodes": 0, "moc": 0}

    with db.tenant_session(tenant_id) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM kg_nodes ORDER BY node_type, label")
            nodes = [dict(r) for r in cur.fetchall()]
            cur.execute("""
                SELECT e.edge_type, s.label AS src, s.node_type AS src_type,
                       d.label AS dst, d.node_type AS dst_type, e.weight
                FROM kg_edges e
                JOIN kg_nodes s ON s.node_id = e.src_id
                JOIN kg_nodes d ON d.node_id = e.dst_id
            """)
            edges = [dict(r) for r in cur.fetchall()]

    by_label = {n["label"]: n for n in nodes}
    out_edges: dict[str, list[dict]] = {}
    in_edges: dict[str, list[dict]] = {}
    for e in edges:
        out_edges.setdefault(e["src"], []).append(e)
        in_edges.setdefault(e["dst"], []).append(e)

    TYPE_ZH = {"company": "企業", "person": "自然人", "document": "文件",
               "jurisdiction": "法域", "period": "期間", "topic": "主題",
               "invoice": "發票", "contract": "合約"}
    EDGE_ZH = {"issued_to": "開票給", "applies_to": "適用於",
               "represented_by": "負責人", "supersedes": "取代",
               "party_to": "當事人"}

    def safe(name: str) -> str:
        return _FS_UNSAFE.sub("_", name)[:80]

    for n in nodes:
        label, ntype = n["label"], n["node_type"]
        attrs = n["attrs"] if isinstance(n["attrs"], dict) else {}
        fm = ["---", f"type: {ntype}", f"type_zh: {TYPE_ZH.get(ntype, ntype)}",
              f"tenant: {n['tenant_id']}"]
        for k, v in attrs.items():
            if v is not None:
                fm.append(f"{k}: {v}")
        fm += [f"out_degree: {len(out_edges.get(label, []))}",
               f"in_degree: {len(in_edges.get(label, []))}", "---", ""]

        body = [f"# {label}", "", f"**類型**：{TYPE_ZH.get(ntype, ntype)}", ""]
        if attrs:
            body += ["## 屬性", ""]
            body += [f"- **{k}**：{v}" for k, v in attrs.items() if v is not None]
            body.append("")
        if out_edges.get(label):
            body += ["## 指向", ""]
            for e in sorted(out_edges[label], key=lambda x: -(x["weight"] or 0))[:60]:
                w = f"（{e['weight']:,.0f}）" if e["weight"] else ""
                body.append(f"- {EDGE_ZH.get(e['edge_type'], e['edge_type'])} "
                            f"→ [[{safe(e['dst'])}]]{w}")
            body.append("")
        if in_edges.get(label):
            body += ["## 被指向", ""]
            for e in sorted(in_edges[label], key=lambda x: -(x["weight"] or 0))[:60]:
                w = f"（{e['weight']:,.0f}）" if e["weight"] else ""
                body.append(f"- [[{safe(e['src'])}]] "
                            f"{EDGE_ZH.get(e['edge_type'], e['edge_type'])} → 本節點{w}")
            body.append("")

        (out_dir / f"{safe(label)}.md").write_text(
            "\n".join(fm + body), encoding="utf-8")
        written["nodes"] += 1

    # ── MOC：每個型別一張目錄頁 + 一張總覽 ────────────────────────────
    by_type: dict[str, list[dict]] = {}
    for n in nodes:
        by_type.setdefault(n["node_type"], []).append(n)

    for ntype, group in by_type.items():
        zh = TYPE_ZH.get(ntype, ntype)
        lines = ["---", "type: MOC", f"covers: {ntype}", "---", "",
                 f"# MOC — {zh}", "",
                 f"共 {len(group)} 個節點。", ""]
        for n in sorted(group, key=lambda x: -(len(out_edges.get(x["label"], []))
                                               + len(in_edges.get(x["label"], [])))):
            deg = len(out_edges.get(n["label"], [])) + len(in_edges.get(n["label"], []))
            lines.append(f"- [[{safe(n['label'])}]]　連結度 {deg}")
        (out_dir / f"MOC-{zh}.md").write_text("\n".join(lines), encoding="utf-8")
        written["moc"] += 1

    index = ["---", "type: MOC", "---", "",
             f"# FlowMind 知識圖譜　（{tenant_id}）", "",
             f"節點 {len(nodes)} 個、關係 {len(edges)} 條。", "",
             "## 依型別瀏覽", ""]
    for ntype, group in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        index.append(f"- [[MOC-{TYPE_ZH.get(ntype, ntype)}]]　{len(group)} 個")
    index += ["", "## 怎麼用", "",
              "1. 開啟 Obsidian 的**關係圖檢視**（Graph View）看整體結構",
              "2. 點任何一個企業節點，看它「開票給」誰、被誰「開票」",
              "3. 自然人節點若連到兩家以上企業，就是**關係企業**的直接證據",
              "4. 文件節點的 `applies_to` 指向法域與期間 ——",
              "   這是回答「這份文件講的是哪個國家、哪一年」的結構化依據", ""]
    (out_dir / "000-知識圖譜總覽.md").write_text("\n".join(index), encoding="utf-8")
    written["moc"] += 1
    return written
