"""
AI 內容工廠 — Vercel Serverless API
使用 Anthropic Claude API
"""
import json
import os
import anthropic
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL = "claude-opus-4-5"

def get_client():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError("Missing ANTHROPIC_API_KEY")
    return anthropic.Anthropic(api_key=key)

# ── Request Models ─────────────────────────────────────────────────────────────
class NewsReq(BaseModel):
    keyword: str = "AI 科技最新動態"

class ContentReq(BaseModel):
    topic: str
    context: Optional[str] = None
    tone: Optional[str] = "輕鬆活潑"
    video_type: Optional[str] = "長影音"
    platforms: Optional[str] = None  # "youtube,instagram,threads"

# ── SSE Stream helper ──────────────────────────────────────────────────────────
def stream_response(system: str, user_msg: str, use_web_search: bool = False):
    def generate():
        try:
            client = get_client()
            kwargs = dict(
                model=MODEL,
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": user_msg}],
            )
            if use_web_search:
                kwargs["tools"] = [
                    {"type": "web_search_20250305", "name": "web_search"},
                ]
            with client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    yield f"data: {json.dumps(text, ensure_ascii=False)}\n\n"
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
    system = """你是科技/AI 新聞研究員，整理最新熱門話題給台灣自媒體創作者。

## 🔥 今日熱門話題（輸出6-8條，繁體中文）

---
### 1. [新聞標題]
**熱度：** ⭐⭐⭐⭐⭐
**摘要：** [2-3句重點]
**爆點：** [為什麼現在很熱？]
**內容潛力：** [腳本/文章/圖文？]
---"""
    return stream_response(
        system,
        f"搜尋「{req.keyword}」最新熱門話題，整理成創作者可用格式。",
        use_web_search=True,
    )

# ── 2. 腳本生產師 ──────────────────────────────────────────────────────────────
@app.post("/api/script")
async def script(req: ContentReq):
    tone = req.tone or "輕鬆活潑"
    ctx = f"\n\n背景資料：\n{req.context}" if req.context else ""

    if req.video_type == "短影音":
        system = f"""你是短影音腳本師，語氣{tone}，專注台灣觀眾。

# ⚡ 60秒短影音腳本

**開場3秒鉤子：** [超震驚的第一句話]
**核心一句話：** [整支影片最重要的觀點]

**重點1：** [15字內]
**重點2：** [15字內]
**重點3：** [15字內]

**結尾CTA：** [關注+留言引導]

**字幕建議：** [每句15字以內，適合手機看]

繁體中文，節奏快，適合 TikTok/Reels/Shorts。"""
        return stream_response(system, f"主題：{req.topic}{ctx}")
    else:
        system = f"""你是科技 YouTube 腳本師，語氣{tone}，專注台灣觀眾。

# 🎬 影片腳本（5-8分鐘）

**預估時長：** X分鐘 ｜ **目標觀眾：** [描述] ｜ **核心訊息：** [一句話]

---
## ⚡ 開場 Hook（0-30秒）
[強力開場，前3秒抓住觀眾]

## 📌 主體
### 第一段：背景/痛點
### 第二段：核心內容（含數據）
### 第三段：實際應用

## 🎯 結尾 CTA
[訂閱+留言+下集預告]

## 📝 SEO
**標題選項（3個）：**
**標籤：**

繁體中文。"""
        return stream_response(system, f"主題：{req.topic}{ctx}")

# ── 3. 文章生產師 ──────────────────────────────────────────────────────────────
@app.post("/api/article")
async def article(req: ContentReq):
    tone = req.tone or "輕鬆易讀"
    ctx = f"\n\n參考資料：\n{req.context}" if req.context else ""
    system = f"""你是科技媒體文章作家，語氣{tone}，適合台灣讀者。

# [文章主標題]
> [副標題]
**閱讀時間：** 約X分鐘

## 前言
## [小標題1]
## [小標題2]
## [小標題3]
## 結論

**SEO：** 主要關鍵字 / Meta Description

繁體中文，適合 Medium/方格子/Vocus。"""
    return stream_response(system, f"主題：{req.topic}{ctx}")

# ── 4. 圖文生產師 ──────────────────────────────────────────────────────────────
PLATFORM_TEMPLATES = {
    "youtube":   "## 📺 YouTube 社群貼文\n[100字以內，附表情符號]",
    "instagram": "## 📸 Instagram\n**圖片建議：** [...]\n**文案：** [150字]\n**Hashtag：** [10個#標籤]",
    "threads":   "## 🧵 Threads\n[200字以內，引發留言互動]",
    "facebook":  "## 📘 Facebook\n[300字，適合分享連結]",
    "dcard":     "## 💬 Dcard\n**板：** [建議板塊]\n**標題：** [吸引點擊的標題]\n**內文：** [500字，朋友聊天口吻]",
    "tiktok":    "## 🎵 TikTok/抖音\n**開場3秒：** [震驚感文字]\n**說明文字：** [50字+熱門標籤]",
}

@app.post("/api/graphic")
async def graphic(req: ContentReq):
    selected = [p.strip() for p in req.platforms.split(",")] if req.platforms else list(PLATFORM_TEMPLATES.keys())
    sections = "\n\n---\n".join(PLATFORM_TEMPLATES[p] for p in selected if p in PLATFORM_TEMPLATES)
    tone = req.tone or "活潑吸睛"
    system = f"""你是社群媒體文案專家，語氣{tone}，只輸出以下選定平台的貼文，繁體中文。

# 🖼️ 圖文貼文

{sections}

---
## 💡 圖片設計建議
[配色、版型、重點文字]"""
    ctx = f"\n\n背景：\n{req.context}" if req.context else ""
    return stream_response(system, f"主題：{req.topic}{ctx}")

# ── 5. PM 管理師 ───────────────────────────────────────────────────────────────
@app.post("/api/pm")
async def pm(req: ContentReq):
    system = """你是資深內容策略 PM，協助台灣科技自媒體規劃策略。

# 📋 PM 策略分析

## 🎯 主題評估（商業/受眾/競爭 各⭐評分）
## 💰 商業潛力（品牌合作/流量/課程）
## 📅 建議發布排程（表格：日期/平台/內容/說明）
## 🔥 差異化建議
## 📊 KPI 目標（7天/30天）
## ✅ 本週行動清單（可勾選）

具體數字，台灣繁體中文。"""
    ctx = f"\n\n資料：\n{req.context}" if req.context else ""
    return stream_response(system, f"主題：{req.topic}{ctx}")

# ── 6. 數據管理師 AI 分析 ──────────────────────────────────────────────────────
@app.post("/api/insight")
async def insight(req: ContentReq):
    system = """你是數據分析師，幫台灣自媒體創作者解讀數據。

# 📊 數據洞察報告

## 📈 表現概覽
## 🌟 亮點發現
## ⚠️ 需注意的地方
## 💡 立即可執行的優化（3條）
## 🎯 下週行動計畫

以數據為基礎，繁體中文。"""
    return stream_response(system, f"分析數據：\n\n{req.topic}\n\n{req.context or ''}")

# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    return {"status": "ok", "model": MODEL, "api_key_set": has_key}
