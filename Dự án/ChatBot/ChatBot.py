import os
import google.generativeai as genai
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd
from dotenv import load_dotenv

# Load API Key từ .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("API key is missing! Please check your .env file.")

genai.configure(api_key=api_key)

# Cấu hình chatbot
generation_config = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "max_output_tokens": 1024,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
)

# Khởi tạo FastAPI
app = FastAPI()

# Cấu hình CORS để cho phép frontend gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Thay đổi thành domain cụ thể nếu deploy
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_frontend():
    """Trả về trang index.html"""
    return FileResponse("static/index.html")

# Đọc dữ liệu từ file CSV
FILE_PATH = "restaurants.csv"
try:
    df = pd.read_csv(FILE_PATH)
    # Chỉ lấy các cột cần thiết và đổi tên nếu cần
    df = df[['name','address', 'cuisines', 'avgRating', 'priceRange', 'timeOpen']].dropna()

except FileNotFoundError:
    raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {FILE_PATH}")
except KeyError as e:
    raise KeyError(f"Các cột bị thiếu trong CSV: {e}")

@app.get("/chatbot")
def chatbot(query: str = Query(..., description="Câu hỏi về nhà hàng")):
    """Chatbot tư vấn nhà hàng"""
    if not query:
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống.")
    
    # Tạo danh sách nhà hàng dạng Markdown
    restaurant_list = "\n".join(
    [
        f"<strong>{row['name']}</strong>\n"
        f"🏠 Địa chỉ: {row['address']}\n"
        f"🍽️ Ẩm thực: {row['cuisines']}\n"
        f"⭐ Đánh giá: {row['avgRating']}/5\n"
        f"💰 Mức giá: {row['priceRange']}\n"
        f"🕒 Giờ mở cửa: {row['timeOpen']}\n"
        for _, row in df.iterrows()
    ]
)

    
    # Tạo lời nhắc cho chatbot
    prompt = f"""
    Bạn là một chatbot tư vấn nhà hàng cho người dùng trên website 
    Người dùng hỏi: '{query}'
    
    Hãy trả lời câu hỏi của người dùng một cách rõ ràng và dễ hiểu.
    với mỗi quán ăn, hãy chia thông tin ra từng dòng.
    mỗi quán ăn sẽ được phân cách bằng dấu xuống dòng. 
    Hãy lựa chọn những nhà hàng phù hợp nhất với câu hỏi của người dùng.
    Nếu không tìm thấy nhà hàng phù hợp, hãy đưa ra những gợi ý ẩm thực khác.
    Định dạng câu trả lời của bạn theo Markdown để dễ đọc.

    Dưới đây là các nhà hàng phù hợp với yêu cầu của bạn
    {restaurant_list}
    Nếu không tìm thấy nhà hàng phù hợp, hãy đưa ra lời gợi ý hợp lý.
    """


    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(prompt)
    
    return {"query": query, "response": response.text}

# Chạy FastAPI trên VS Code
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
