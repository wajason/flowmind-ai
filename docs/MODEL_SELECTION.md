# 模型選型：實測數據與判讀

> 這份文件記錄的是**實際跑出來的數字**，不是規格表比較。
> 所有數據都可以用 `scripts/eval_models.py` 與 `scripts/run_verifin.py` 重跑驗證。

**硬體**：Windows 11 · RTX 4060 Laptop（8GB VRAM）· 32GB RAM · Ollama 原生 Windows 版
**日期**：2026-08-07

---

## 結論先講

| 角色 | 選用 | 一句話理由 |
|---|---|---|
| 文件抽取 | **`qwen3.5:9b`** | HPES 主指標勝出三倍 —— 它比較願意承認不知道 |
| 顧問問答 | **`qwen3.5:9b`** | 與抽取共用同一模型，避免顯存換入換出的 30~60 秒代價 |
| 離線知識合成 | `gpt-oss:20b` | 一次性跑，慢無妨，要的是深度 |
| Embedding | `bge-m3` | 多語言、中英混雜穩定、1024 維、與既有資產同源 |

---

## 這次選型最重要的發現：傳統準確率會讓你選錯模型

在 SROIE 測試集的同一批 12 份文件、48 個欄位上：

| 模型 | 答對 | 答錯 | 留白 | **傳統準確率** | **HPES (λ=2)** |
|---|---|---|---|---|---|
| `gemma4:e4b` | 32 | 15 | **1** | **0.667** ← 較高 | **+0.042** |
| `qwen3.5:9b` | 30 | 12 | **6** | 0.625 | **+0.125** ← 較高 |

**如果只看準確率，會選 gemma4:e4b。而那是錯的選擇。**

原因在留白次數：gemma4 在 48 個欄位裡只留白 1 次，
也就是它幾乎「一定要填點什麼」，代價是猜錯 15 次。
qwen3.5 留白 6 次，猜錯只有 12 次。

在授信送件的場域，這兩種錯誤的成本天差地遠：

- 一個**空欄位** → 銀行要求補件，多花兩天
- 一個**編造的買方統一編號** → 整份申請被退回，嚴重時客戶被列入警示

HPES 用 λ=2 把這個成本差距寫進計分（答錯 −2、留白 0、答對 +1），
於是它選出了在真實場域裡實際比較好用的那個模型。
這正是我們設計這個指標的原因 —— 不是為了讓分數好看，
是因為既有指標會系統性地獎勵「勇於亂猜」的模型。

其餘指標兩者接近，沒有改變結論：

| 指標 | gemma4:e4b | qwen3.5:9b |
|---|---|---|
| 引用可驗證率 CVR | 97.8% | 97.5% |
| 憑空生成率（值在原文中不存在） | 2.13% | 2.38% |
| AURC（越低越好） | 0.244 | 0.241 |
| 嚴格 JSON 失敗率 | 0.0% | 0.0% |

---

## 一個中途翻案：qwen3.5:9b 一開始被判定為不可用

第一輪用 `scripts/eval_models.py` 的簡易測試時，結果是：

```
qwen3.5:9b   T1 抽取正確率 0%   嚴格JSON=False
gemma4:e4b   T1 抽取正確率 100%  嚴格JSON=True
```

看起來 qwen3.5 完全不能用。但根因不是模型能力，是**呼叫方式**：
qwen3.5 是 thinking 類模型，會先輸出 `<think>…</think>` 推理段落，
在 JSON 模式下直接破格。

修正方式寫在 [`flowmind/llm.py`](../flowmind/llm.py) 的 `extract_json()`：

1. 帶 `think: false` 關掉推理輸出（並在舊版 Ollama 回 400 時自動退回重試）
2. 把 `format` 從 `"json"` 改成完整的 **JSON Schema** —— 這是
   grammar-constrained decoding，解碼時直接把不合法的 token 機率壓成 0
3. 統一剝除殘留的 thinking 標籤

修正後 qwen3.5 的嚴格 JSON 失敗率是 **0.0%**，而且在主指標上反超。

**這件事本身值得記錄**：一個模型「不能用」的結論，
有很高機率是呼叫方式的問題而不是模型的問題。
如果當時就據此把 qwen3.5 排除，我們會選到一個在真實場域比較差的模型。

---

## 顯存：8GB 是這台機器上真正的約束

| 模型 | 磁碟大小 | 8GB VRAM 下的實際行為 |
|---|---|---|
| `qwen3.5:9b` | 6.6 GB | ✅ 完整放進顯存 |
| `gemma4:e4b` | 9.6 GB | ⚠️ 必定溢出，部分層跑 CPU |
| `gpt-oss:20b` | 13 GB | ⚠️ 大量溢出，僅適合離線批次 |
| `qwen3.6:35b` | 23 GB | ❌ 互動式使用不可行 |

實測時 `ollama ps` 顯示過 `23%/77% CPU/GPU` 的分流狀態
（當時 embedding 模型與 LLM 同時常駐）。
Ollama 遇到顯存不足**不會報錯**，只會靜默退回部分 CPU 推論 ——
表現是「突然慢 5~8 倍而且不知道為什麼」。

這直接導出兩個設計決定：

1. **Embedding 走 Ollama 而不是 sentence-transformers。**
   sentence-transformers 載入 bge-m3 會固定佔約 2.4GB 顯存，
   跟 LLM 相加就爆了。交給 Ollama 統一排程，顯存管理變成它的問題。
   副作用：這個專案完全不需要安裝 PyTorch。

2. **抽取與顧問用同一個模型。**
   用兩個不同模型會讓 Ollama 反覆換入換出，每次付 30~60 秒載入成本。
   在決賽 demo 現場，這種延遲是會失分的。

### 速度

| 模型 | SROIE 每份文件耗時 | 備註 |
|---|---|---|
| `gemma4:e4b` | 約 38 秒 | 12 份 / 454 秒 |
| `qwen3.5:9b` | 約 44 秒 | 12 份 / 527 秒，但**測試期間有 embedding 模型同時佔用顯存**，非公平比較 |

速度數字暫時不作為選型依據 —— 兩次測試的顯存競爭條件不同。
正式部署到雲端 API 後這個變數會消失。

---

## 為什麼保留 LiteLLM 而不直接綁 Ollama SDK

換成 Claude API 或 Azure OpenAI 只需要改 `.env` 三行，程式碼零修改：

```env
LITELLM_BASE_URL=https://api.anthropic.com/v1
LITELLM_API_KEY=sk-ant-...
LLM_ADVISOR_MODEL=claude-sonnet-5
```

對一個要進企業 POC 的產品，被單一模型供應商綁死是實質風險。
本地 Ollama 的價值在於「資料不出本機」——
中小企業的發票、合約、銀行流水屬於營業秘密，
這不是技術偏好，是這個場域能不能被採用的前提。
但這個價值不應該以「換不掉」為代價。

---

## 重跑這些數據

```powershell
# 快速三項測試（抽取遵從度 / 拒答紀律 / 中文金融語感）
python scripts\eval_models.py --models qwen3.5:9b gemma4:e4b

# VeriFin 完整評測（含反事實擾動）
python scripts\run_verifin.py --suite sroie --limit 12 --model qwen3.5:9b
python scripts\run_verifin.py --suite sroie --limit 12 --model gemma4:e4b

# 全量正式數據（耗時較久）
python scripts\run_verifin.py --suite all --limit 0 --counterfactual
```

> ⚠️ 上表數字取自 `--limit 12` 的小樣本，僅足以支撐「兩個模型的相對排序」，
> 不足以宣稱絕對準確率。正式提案書引用的數字應以 `--limit 0` 全量重跑為準。
> 這裡把樣本數寫出來，是因為一個沒有標示樣本數的準確率沒有意義。
