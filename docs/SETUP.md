# 環境建置與執行

Linux 與 Windows 都能跑，以下兩套指令等價，選一套照做即可。
後續章節的指令以 Linux 為主，
Windows 使用者把 `python` 換成 `.venv\Scripts\python.exe`、
路徑分隔改成 `\` 即可。

| | Linux / macOS | Windows |
|---|---|---|
| Shell | bash / zsh | PowerShell |
| 虛擬環境啟動 | `source .venv/bin/activate` | `.venv\Scripts\Activate.ps1` |
| 路徑分隔 | `/` | `\` |
| Docker | Docker Engine | Docker Desktop |

## 1. 前置需求

| 元件 | 版本 | 為什麼需要 |
|---|---|---|
| Python | 3.11 | 用 `uv` 管理，不動系統既有 Python |
| Docker + Compose | 任意近期版本 | 只跑 PostgreSQL + pgvector |
| Ollama | 任意近期版本 | 本地推論；金融場域不外送資料 |
| Git | 任意 | — |

**硬體**：本專案的效能數字量測自 8GB VRAM 的環境。
`gemma4:26b`（17GB）在 8GB 顯存上會有 CPU/GPU 分流，
冷啟動約 124 秒、熱啟動約 9 秒。
16GB 以上顯存可完全載入，速度明顯較快。
**顯存不足不影響正確性，只影響速度** —— 所有正確性測試都不依賴 GPU。

## 2. 一次性建置

<details open>
<summary><b>Linux / macOS</b></summary>

```bash
# ① 取得程式碼（放哪裡都可以，以下用 ~/flowmind-ai 為例）
git clone https://github.com/wajason/flowmind-ai.git ~/flowmind-ai
cd ~/flowmind-ai

# ② 建立虛擬環境（uv：https://docs.astral.sh/uv/）
curl -LsSf https://astral.sh/uv/install.sh | sh      # 若尚未安裝 uv
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt

# ③ 下載模型（約 18GB，一次即可）
ollama pull gemma4:26b      # 抽取 + 顧問
ollama pull bge-m3          # Embedding（1024 維）

# ④ 啟動向量資料庫（host port 5433，刻意避開系統既有的 5432）
docker compose up -d
docker compose ps           # 應顯示 healthy

# ⑤ 設定環境變數
cp .env.example .env

# ⑥ 環境自檢 —— 缺什麼它會直接告訴你
python -m flowmind.cli doctor
```

</details>

<details>
<summary><b>Windows（PowerShell）</b></summary>

```powershell
# ① 取得程式碼
git clone https://github.com/wajason/flowmind-ai.git flowmind-ai
cd flowmind-ai

# ② 建立虛擬環境
winget install --id=astral-sh.uv -e                  # 若尚未安裝 uv
uv venv .venv --python 3.11
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt

# ③ 下載模型（約 18GB，一次即可）
ollama pull gemma4:26b
ollama pull bge-m3

# ④ 啟動向量資料庫
docker compose up -d
docker compose ps

# ⑤ 設定環境變數
Copy-Item .env.example .env

# ⑥ 環境自檢
python -m flowmind.cli doctor
```

> **PowerShell 執行原則**：若 `Activate.ps1` 被擋，執行一次
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`。

</details>

`doctor` 全綠的樣子：

```
✅ Ollama 連線正常（14 個模型）
   ✅ 抽取模型 gemma4:26b   ✅ 向量模型 bge-m3
✅ PostgreSQL 連線正常（pgvector 0.8.6），SHARED 知識庫 6889 個 chunk
✅ 連線角色 flowmind_app 非 superuser，Row-Level Security 生效
✅ 稽核軌跡 10 筆，雜湊鏈完整
```

> **為什麼是 port 5433**：刻意避開 5432。你的電腦上很可能已有其他專案的
> PostgreSQL。`docker compose up` 撞 port 而失敗，是最沒必要的麻煩。
>
> **為什麼用 `flowmind_app` 而不是 `flowmind` 連線**：後者是 superuser，
> **會繞過 Row-Level Security**，隔離形同虛設。`doctor` 會檢查這一項。

## 3. 建立資料

```powershell
# 公開知識庫（法規 · 信保基金要點 · 銀行商品說明）
python scripts\fetch_public_corpus.py
python data_update_finance.py --tenant SHARED --rebuild     # 約 40 分鐘（含大型白皮書）

# 真實企業交易資料（政府採購決標公告：真實統編/金額/日期）
python scripts\fetch_real_corpus.py --source pcc --industry-preset manufacturing --pages 2

# 示範委任案
python generate_synthetic_data.py --seed 42 --outdir data\raw\CASE-0001
python data_update_finance.py --tenant CASE-0001 --rebuild

# 負向對照組（刻意注入五項已知缺陷）
python generate_synthetic_data.py --seed 7 --inject-defects --outdir data\raw\CASE-9999
python data_update_finance.py --tenant CASE-9999 --rebuild
```

## 4. 日常使用

```powershell
# 顧問問答（含證據包輸出）
python rag_query.py --tenant CASE-0001 -q "信保基金供應商融資的保證成數最高幾成？"
python rag_query.py --tenant CASE-0001                    # 互動模式
python rag_query.py --tenant CASE-0001 -q "…" --json      # 給下游系統串接
python rag_query.py --tenant CASE-0001 -q "…" --force-rag # 略過決定性路由做對照

# 決定性交叉驗證（零 LLM）
python -m flowmind.cli crosscheck --tenant CASE-0001
python -m flowmind.cli crosscheck --tenant CASE-9999 --against-answer-key

# 資料隔離與稽核證明
python rag_query.py --verify-isolation CASE-0001 CASE-9999
python rag_query.py --verify-audit
python -m flowmind.cli engagements
```

## 5. 評測與測試

```powershell
# 回歸測試（201 項，數秒，不需資料庫與 LLM）
python tests\test_core.py

# 外部 benchmark
python scripts\fetch_benchmarks.py                        # SROIE / FUNSD / CORD
python scripts\run_verifin.py --suite sroie --limit 50
python scripts\run_verifin.py --suite all --limit 0 --counterfactual   # 正式數據

# 模型選型實測（6 模型 × 5 面向完整矩陣，約 25 分鐘）
python scripts\model_matrix.py
python scripts\model_matrix.py --models gemma4:26b qwen3.6:35b   # 只比特定模型
```

## 6. 產生領域技能檔

```powershell
python skill_builder.py --tenant SHARED
# → out/skills/taiwan-supply-chain-finance/SKILL.md
```

產出符合 **Agent Skills 開放標準**（YAML frontmatter + Markdown），
可放進 `.claude/skills/`、也可餵給 Hermes / llama.cpp / Ollama 等任何執行環境
（見該目錄下的 `PORTING.md`）。

> ⚠️ 目前 skill_builder 的引用驗證率僅 **31.8%**，
> **人工覆核前不得對外交付**。原因與改善方向見 [MODEL_SELECTION.md](MODEL_SELECTION.md)。

---
