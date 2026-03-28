"""
AI 內容工廠 — Vercel Serverless API
使用 Google Gemini API（免費方案）
"""
import json
import os
import google.generativeai as genai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL = "gemini-1.5-flash"

def get_model(system: str, use_web_search: bool = False):
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise ValueError("Missing GEMINI_API_KEY")
    genai.configure(api_key=key)
    kwargs = {
        "model_name": MODEL,
        "system_instruction": system,
    }
    if use_web_search:
        kwargs["tools"] = "google_search_retrieval"
    return genai.GenerativeModel(**kwargs)

# ── Request Models ─────────────────────────────────────────────────────────────
class NewsReq(BaseModel):
    keyword: str = "AI 科技最新動態"

class ContentReq(BaseModel):
    topic: str
    context: Optional[str] = None

# ── SSE Stream helper ──────────────────────────────────────────────────────────
def stream_response(system: str, user_msg: str, use_web_search: bool = False):
    def generate():
        try:
            model = get_model(system, use_web_search)
            response = model.generate_content(user_msg, stream=True)
            for chunk in response:
                if chunk.text:
                    yield f"data: {json.dumps(chunk.text, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps('❌ ' + str(e))}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

# ── 1. 找新聞的 ────────────────────────────────────────────────────────────────
@app.post("/api/news")
async def news(req: NewsReq):
    system = """你是一位專業的科技/AI 新聞研究員。任務是搜尋並整理最新的熱門話題給台灣自媒體創作者使用。

請以下列格式輸出 6-8 條新聞（繁體中文）：

## 🔥 今日熱門話題

---
### 1. [新聞標題]
**熱度：** ⭐⭐⭐⭐⭐
**摘要：** [2-3句重點]
**爆點：** [為什麼現在很熱？]
**內容潛力：** [適合腳本/文章/圖文？]
---"""
    return stream_response(
        system,
        f"請搜尋「{req.keyword}」的最新熱門話題，整理成創作者可用的格式。",
        use_web_search=True,
    )

# ── 2. 腳本生產師 ──────────────────────────────────────────────────────────────
@app.post("/api/script")
async def script(req: ContentReq):
    system = """你是專業科技 YouTube 腳本師，擅長把 AI/科技話題變成讓觀眾欲罷不能的影片腳本。

請生成完整腳本：

# 🎬 影片腳本

**預估時長：** X-X 分鐘
**目標觀眾：** [描述]
**核心訊息：** [一句話]

---

## ⚡ 開場 Hook（0-30秒）
[強力開場，前3秒抓住觀眾]

---

## 📌 主體內容

### 第一段：背景/痛點
### 第二段：核心內容（含數據、案例）
### 第三段：實際應用

---

## 🎯 結尾 CTA
[訂閱提醒、留言引導、下集預告]

---

## 📝 SEO
**標題選項（3個）：**
**標籤：**

台灣繁體中文，活潑有個性的風格。"""
    ctx = f"\n\n背景資料：\n{req.context}" if req.context else ""
    return stream_response(system, f"請為此主題製作完整影片腳本：{req.topic}{ctx}")

# ── 3. 文章生產師 ──────────────────────────────────────────────────────────────
@app.post("/api/article")
async def article(req: ContentReq):
    system = """你是專業科技媒體文章作家，把 AI/科技話題寫成既深度又易讀的文章，適合台灣讀者。

# [文章主標題]

> [副標題：一句話說明核心價值]

**閱讀時間：** 約X分鐘

---

## 前言
[引人入勝的開頭]

## [小標題1]
[詳細內容，含數據、引用]

## [小標題2]

## [小標題3]

## 結論：對你的影響
[實際建議]

---

**SEO 優化：**
- 主要關鍵字：
- Meta Description：

台灣繁體中文，適合 Medium/方格子/Vocus 發布。"""
    ctx = f"\n\n參考資料：\n{req.context}" if req.context else ""
    return stream_response(system, f"請為此主題撰寫完整深度文章：{req.topic}{ctx}")

# ── 4. 圖文生產師 ──────────────────────────────────────────────────────────────
@app.post("/api/graphic")
async def graphic(req: ContentReq):
    system = """你是社群媒體文案專家，為各平台製作吸引人的圖文貼文。

# 🖼️ 多平台圖文貼文

---
## 📺 YouTube 社群貼文
[100字以內]

---
## 📸 Instagram
**圖片建議：** [...]
**貼文文案：** [150-200字]
**Hashtag：** #[...] #[...]（10-15個）

---
## 🧵 Threads
[200字以內，引發互動]

---
## 📘 Facebook
[300字，可分享連結]

---
## 💬 Dcard
**板：** [建議板塊]
**標題：** [吸引點擊]
**內文：** [500-800字，像朋友聊天的口吻]

---
## 🎵 TikTok
**開場3秒文字：** [震驚感]
**說明文字：** [50字]

---
## 💡 圖片設計建議
[配色、版型、重點文字]

台灣繁體中文，各平台符合該平台用戶習慣。"""
    ctx = f"\n\n背景資料：\n{req.context}" if req.context else ""
    return stream_response(system, f"請為此主題製作各平台圖文貼文：{req.topic}{ctx}")

# ── 5. PM 管理師 ───────────────────────────────────────────────────────────────
@app.post("/api/pm")
async def pm(req: ContentReq):
    system = """你是資深內容策略 PM，協助科技自媒體規劃策略、評估商業價值。

# 📋 PM 策略分析報告

---
## 🎯 主題評估
**商業價值：** ⭐⭐⭐⭐⭐ (X/5)
**受眾吸引力：** ⭐⭐⭐⭐⭐ (X/5)
**競爭程度：** [低/中/高]

---
## 💰 商業潛力
### 品牌合作機會
### 流量變現
### 課程/知識變現

---
## 📅 建議發布排程
| 日期 | 平台 | 內容 | 說明 |
|------|------|------|------|
| Day 1 | YouTube | 完整影片 | ... |

---
## 🔥 差異化建議
[競品分析 + 獨特切角]

---
## 📊 KPI 目標
- 7天目標：
- 30天目標：

---
## ✅ 行動清單（本週）
- [ ] [任務1]
- [ ] [任務2]

具體數字和可執行建議，台灣繁體中文。"""
    ctx = f"\n\n相關資料：\n{req.context}" if req.context else ""
    return stream_response(system, f"請為此主題提供完整 PM 策略分析：{req.topic}{ctx}")

# ── 6. 數據管理師 AI 分析 ──────────────────────────────────────────────────────
@app.post("/api/insight")
async def insight(req: ContentReq):
    system = """你是數據分析師，幫助自媒體創作者解讀內容數據，找出成長機會。

# 📊 數據洞察報告

---
## 📈 表現概覽
## 🌟 亮點發現
## ⚠️ 需注意的地方
## 💡 優化建議（可立即執行）
1. [具體建議]
2. [具體建議]
3. [具體建議]
## 🎯 下週行動計畫
## 📅 追蹤建議

以數據為基礎，台灣繁體中文。"""
    return stream_response(system, f"請分析以下數據：\n\n{req.topic}\n\n{req.context or ''}")

# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    has_key = bool(os.environ.get("GEMINI_API_KEY"))
    return {"status": "ok", "model": MODEL, "api_key_set": has_key}
