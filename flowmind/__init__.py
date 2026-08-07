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
