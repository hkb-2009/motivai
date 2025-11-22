import os
import re
from typing import Optional

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Gemini
import google.generativeai as genai

# ---------- Config & bootstrap ----------
load_dotenv()  # đọc .env nếu có

key = input("Enter Key: ")

GEMINI_API_KEY = os.getenv(key, "").strip()
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
# Block một số nội dung nguy cơ cao (bạn có thể mở rộng)
BLOCKLIST = re.compile(
    r"(suicide|tự\s*sát|ma\s*túy|phishing|carding|hack\s*\*?ai)",
    re.IGNORECASE,
)

# Nếu có API key thì khởi tạo model
MODEL = None
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    MODEL = genai.GenerativeModel(
        MODEL_NAME,
        generation_config={
            "temperature": 0.522,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        },
        safety_settings=None,  # dùng mặc định của Google; có thể tuỳ chỉnh theo chính sách
    )


# ---------- Helpers ----------
def stub_reply(msg: str) -> str:
    """Trả lời tạm khi chưa có API key (để dev/test nhanh)."""
    trimmed = (msg or "").strip()
    if len(trimmed) > 100:
        trimmed = trimmed[:100] + "…"
    return (
        f"(MOTIVAI–stub) Mình đã nhận mục tiêu của bạn: “{trimmed}”. "
        "Bắt đầu bằng 1 bước nhỏ ngay hôm nay nhé! 💪"
    )

'''
def motivate_users(msg: str) -> str:
    prompt = f"""
        Bạn là MOTIVAI – trợ lý AI hỗ trợ thay đổi hành vi và xây dựng thói quen lành mạnh
        
        NGUYÊN TẮC HOẠT ĐỘNG:
        1) Nếu đây là lần đầu người dùng nhắc tới một vấn đề mới 
           (ví dụ: cai nghiện, học tập, sức khỏe, cảm xúc...) 
           và thông tin còn chung chung, 
           HÃY:
           
        CHỈ đặt 2–3 câu hỏi làm rõ (ngắn, dễ trả lời).
        Không tư vấn sâu, chỉ nói 1 câu ngắn kiểu “Để giúp bạn tốt hơn mình hỏi nhanh vài ý…”
        
        2) Nếu người dùng đã cung cấp khá nhiều thông tin về cùng một vấn đề 
           (đã trả lời các câu hỏi trước đó):
           
        Bắt đầu bằng 1–2 câu tóm tắt lại bối cảnh của họ
        Sau đó đưa ra gợi ý / kế hoạch hành động cụ thể theo từng bước.
        Kết thúc bằng 1 câu động viên rõ ràng, dễ thực hiện ngay hôm nay
        
        3) Luôn dùng giọng văn:
           
        Tôn trọng, không phán xét.
        Tích cực, thực tế, không “chữa lành” sáo rỗng.
        Câu ngắn, dễ đọc trên màn hình điện thoại.
        
        Tin nhắn người dùng:
        "{message}"
        
        Hãy trả lời đúng theo NGUYÊN TẮC HOẠT ĐỘNG ở trên.
        """
    response = model.generate_content(prompt)
    return response.text.strip()
''' ## APPARENTLY THIS FUNCTION IS NEVER USED ##

def build_system_prompt(category: Optional[str]) -> str:
    """
    Prompt hệ thống nhẹ, điều chỉnh giọng điệu theo nhóm.
    category ∈ {'habit','study','emotion'} nếu frontend gửi.
    """
    base = (
        """
            Bạn là MOTIVAI – trợ lý AI hỗ trợ thay đổi hành vi và xây dựng thói quen lành mạnh

            NGUYÊN TẮC HOẠT ĐỘNG:
            1) Nếu đây là lần đầu người dùng nhắc tới một vấn đề mới 
               (ví dụ: cai nghiện, học tập, sức khỏe, cảm xúc...) 
               và thông tin còn chung chung, 
               HÃY:

            CHỈ đặt 2–3 câu hỏi làm rõ (ngắn, dễ trả lời).
            Không tư vấn sâu, chỉ nói 1 câu ngắn kiểu “Để giúp bạn tốt hơn mình hỏi nhanh vài ý…”

            2) Nếu người dùng đã cung cấp khá nhiều thông tin về cùng một vấn đề 
               (đã trả lời các câu hỏi trước đó):

            Bắt đầu bằng 1–2 câu tóm tắt lại bối cảnh của họ
            Sau đó đưa ra gợi ý / kế hoạch hành động cụ thể theo từng bước.
            Kết thúc bằng 1 câu động viên rõ ràng, dễ thực hiện ngay hôm nay

            3) Luôn dùng giọng văn:

            Tôn trọng, không phán xét.
            Tích cực, thực tế, không “chữa lành” sáo rỗng.
            Câu ngắn, dễ đọc trên màn hình điện thoại.

            Tin nhắn người dùng:
            "{message}"

            Hãy trả lời đúng theo NGUYÊN TẮC HOẠT ĐỘNG ở trên.
        """
        "You are MOTIVAI, a concise, upbeat motivation coach. "
        "Always be practical, non-judgmental, and action-oriented. "
        "Give a solution, a roadmap to help with the problem. "
        "Write 2–5 short bullet points max, in Vietnamese, with 1 emoji at the end.\n"
        "Reponse in relation with the question. "
    )
    if category == "habit":
        base += "Focus on tiny habits, triggers, and 1 next action in under 30 seconds.\n"
    elif category == "study":
        base += "Focus on time-blocks, distraction control, and a 25–50 minute plan.\n"
    elif category == "emotion":
        base += "Acknowledge feelings, suggest one grounding technique, and a small step.\n"
    return base


def call_gemini(user_message: str, category: Optional[str], history: list[dict]) -> str:
    """Gọi Gemini với lịch sử chat và system instruction chính xác."""
    sys_instruction = build_system_prompt(category)
    current_model = genai.GenerativeModel(
        MODEL_NAME,
        system_instruction=sys_instruction,
        generation_config={
            "temperature": 0.522,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
    )

    gemini_history = []
    for turn in history:
        if turn.get("role") in ["user", "model"] and turn.get("parts"):
            gemini_history.append({
                "role": turn["role"],
                "parts": turn["parts"]
            })

    chat_session = current_model.start_chat(history=gemini_history)

    try:
        resp = chat_session.send_message(user_message)

        text = getattr(resp, "text", "") or ""
        text = text.replace("[LINEBREAK]", "\n\n")
        return text.strip()
    except Exception as e:
        print("Error Message inside call_gemini:", e)
        return ""


# ---------- Routes ----------
@app.get("/health")
def health():
    return jsonify(
        {
            "ok": True,
            "service": "motivai-backend",
            "model": MODEL_NAME,
            "gemini_configured": bool(MODEL),
        }
    )


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    category = (data.get("category") or "").strip().lower() or None

    history = data.get("history") or []

    # Validate
    if not message or len(message) > 2000:
        return jsonify(error="message invalid or too long"), 400
    if BLOCKLIST.search(message):
        return jsonify(error="topic not supported"), 400

    # Nếu chưa cấu hình API key -> stub
    if not GEMINI_API_KEY:
        return jsonify(reply=stub_reply(message), mode="stub"), 200

    try:
        reply = call_gemini(message, category, history)
        if not reply:
            return jsonify(reply="Mình đang gặp chút sự cố, thử lại giúp mình nhé!", mode="error"), 200
        return jsonify(reply=reply, mode="gemini"), 200
    except Exception as e:
        print("Error Message:", e)
        # fallback an toàn
        return jsonify(reply=stub_reply(message), mode="fallback", detail=str(e)), 200

# ---------- Entrypoint ----------
if __name__ == "__main__":
    # Chạy local: python backend/app.py
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("DEBUG", "false") == "true")