# SKILL.md 跨平台使用指南

`SKILL.md` 遵循 **Agent Skills 開放標準**（YAML frontmatter + Markdown），
由 Anthropic 於 2025 年提出並開放，目前多個執行環境支援。

**但這份檔案本質上就是一份純文字的領域知識**，
所以即使在不支援該標準的環境，也能直接當 system prompt 使用。
這一點很重要：它讓這份產出可以交付給合作銀行，
對方**不需要採用我們整套系統**就能先拿到價值。

---

## 1. Claude Code / Claude.ai

```powershell
# 專案層級
mkdir .claude\skills\taiwan-supply-chain-finance
copy skills\taiwan-supply-chain-finance\SKILL.md .claude\skills\taiwan-supply-chain-finance\

# 使用者層級（所有專案共用）
copy skills\taiwan-supply-chain-finance\SKILL.md %USERPROFILE%\.claude\skills\taiwan-supply-chain-finance\
```

Claude 會依 frontmatter 的 `description` 自動判斷何時載入。

---

## 2. Ollama（本地）

### 方式 A：Modelfile（做成一個常駐的專用模型）

```dockerfile
# Modelfile
FROM qwen3.5:9b
PARAMETER temperature 0.2
PARAMETER num_ctx 16384
SYSTEM """
<把 SKILL.md 的內容貼在這裡（去掉 YAML frontmatter）>
"""
```

```powershell
ollama create scf-advisor -f Modelfile
ollama run scf-advisor "信保基金供應商融資的保證成數最高幾成？"
```

### 方式 B：每次呼叫時注入（適合程式整合）

```python
import httpx, pathlib, re

raw = pathlib.Path("skills/taiwan-supply-chain-finance/SKILL.md").read_text(encoding="utf-8")
system = re.sub(r"^---.*?---\s*", "", raw, flags=re.DOTALL)   # 去掉 frontmatter

httpx.post("http://localhost:11434/api/chat", json={
    "model": "qwen3.5:9b",
    "messages": [{"role": "system", "content": system},
                 {"role": "user", "content": "保證成數最高幾成？"}],
    "options": {"num_ctx": 16384},
    "stream": False,
}, timeout=300)
```

> ⚠️ **`num_ctx` 一定要設。** Ollama 的 OpenAI 相容端點無法傳這個參數，
> 預設 context 可能只有 4096，SKILL.md 加上使用者問題會被**靜默截斷** ——
> API 仍回 200，模型卻回空內容或胡言亂語。這個坑我們實際踩過。

---

## 3. llama.cpp

```bash
# 先去掉 YAML frontmatter
sed '1{/^---$/!q};1,/^---$/d' SKILL.md > skill_body.md

./llama-cli -m model.gguf \
  --system-prompt-file skill_body.md \
  --ctx-size 16384 \
  -p "信保基金供應商融資的保證成數最高幾成？"
```

server 模式：

```bash
./llama-server -m model.gguf --ctx-size 16384
# 呼叫時把 skill_body.md 內容放進 messages[0] 的 system role
```

---

## 4. Hermes / 自建 Agent

```python
from pathlib import Path
import re

def load_skill(name: str) -> str:
    """讀取 skill 內容並去掉 frontmatter，回傳可直接當 system prompt 的文字。"""
    p = Path("skills") / name / "SKILL.md"
    raw = p.read_text(encoding="utf-8")
    return re.sub(r"^---.*?---\s*", "", raw, flags=re.DOTALL)

SYSTEM = load_skill("taiwan-supply-chain-finance")
```

若你的 agent 有多個 skill，建議只在**判斷相關時才載入** ——
frontmatter 的 `description` 就是為了讓 router 做這個判斷而存在。

---

## 5. OpenAI Codex · Cursor · Gemini CLI · Windsurf

這些環境支援 Agent Skills 標準或有等價的 rules/instructions 機制，
放置位置各異，但共通做法是：把 `SKILL.md` 放進該工具的
skill／rules／instructions 目錄即可。

---

## 6. 檢查清單：換環境後要驗證什麼

| # | 檢查 | 怎麼確認 |
|---|---|---|
| 1 | **context 夠不夠** | 問一個需要引用文件末段內容的問題，看它答不答得出來 |
| 2 | **拒答行為還在嗎** | 問「2027 年的保證成數是多少」，正確行為是說不知道 |
| 3 | **不會做授信決策** | 問「這案一定會過吧」，正確行為是說明這超出邊界 |
| 4 | **境外問題會拒答** | 問「日本的保證成數是多少」，正確行為是說本技能只涵蓋台灣 |
| 5 | **數字有標出處** | 回答含具體數字時應標註 📗 原文 / 📊 實測 / 🧠 推論 |

> 第 2–4 項特別重要：**skill 的價值有一半在「它不會亂說什麼」。**
> 換環境後若拒答行為消失，代表 system prompt 沒有正確載入，
> 或 context 被截斷了。

---

## 7. 這份 skill 與 FlowMind 系統的關係

| | SKILL.md 單獨使用 | 接上 FlowMind 系統 |
|---|---|---|
| 領域知識 | ✅ 有 | ✅ 有 |
| 引用逐字驗證 | ❌ 靠模型自律 | ✅ 程式回原文比對 |
| 決定性交叉驗證 | ❌ | ✅ 12 項純算術檢查 |
| 客戶資料隔離 | ❌ | ✅ PostgreSQL RLS |
| 稽核軌跡 | ❌ | ✅ 雜湊鏈 |
| 統計數字精確查詢 | ❌ 只有摘要 | ✅ 直接讀原始檔案 |

**單獨使用 SKILL.md 是「懂這個領域的顧問」；
接上系統才是「可驗證的證據層」。**

交付給合作對象時，SKILL.md 是低門檻的第一步，
它能讓對方先感受到領域知識的價值，再談完整系統的導入。
