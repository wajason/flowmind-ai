"""
FlowMind AI — Verifiable Credit Evidence Layer for Supply Chain Finance
=============================================================================
把中小企業的非結構化營運資料（發票、合約、銀行流水、往來 Email），
轉換成「銀行授信人員可以逐項驗證」的證據包。

模組地圖：
    config      單一設定來源
    textnorm    中文 bigram 分詞 + 統編檢核（決定性運算）
    db          RLS 感知連線、稽核鏈、隔離證明
    embeddings  可抽換的向量化後端（Ollama / sentence-transformers）
    llm         角色分工的 LLM 呼叫 + 受約束 JSON 解碼
    retrieval   Hybrid Search（Dense + CJK Sparse）+ RRF + 多樣性過濾
    evidence    Evidence / Confidence / Source / Reason 輸出契約
    crosscheck  發票↔合約↔銀行流水 的決定性交叉驗證
"""

__version__ = "0.4.0"

# ── 強制 UTF-8 輸出（單點修正，涵蓋所有 import flowmind 的腳本）──────────
#
# Windows 的 Python 預設用系統 ANSI 代碼頁寫 stdout
# （繁中環境 cp950、英文環境 cp1252）。本專案的輸出全是中文與框線字元，
# 在那種環境下第一個 print 就會炸：
#     UnicodeEncodeError: 'charmap' codec can't encode characters
#
# 這是 GitHub Actions 的 Windows job 抓到的 —— Linux job 全部通過。
# 本機看不到，是因為開發用的 PowerShell 主控台剛好是 UTF-8。
#
# 修在這裡而不是逐一改 24 支腳本：它們全都 `import flowmind`，
# 一個地方修好，全部跟著好；日後新增的腳本也自動涵蓋。
# 對一個宣稱「Windows 與 Linux 都能跑」的專案來說，
# **工具自己跑不起來是最難堪的一種不可攜。**
def _force_utf8_stdio() -> None:
    import sys                                          # noqa: PLC0415
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:                           # noqa: BLE001
                pass                                    # 已重導向到檔案等情況


_force_utf8_stdio()
