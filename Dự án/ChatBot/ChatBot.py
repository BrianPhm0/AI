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
FILE_PATH = "sorted_restaurants.csv"
try:
    df = pd.read_csv(FILE_PATH)
    # Chỉ lấy các cột cần thiết và đổi tên nếu cần
    df = df[['name','address', 'cuisines', 'average_score', 'priceRange', 'timeOpen']].dropna()

except FileNotFoundError:
    raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {FILE_PATH}")
except KeyError as e:
    raise KeyError(f"Các cột bị thiếu trong CSV: {e}")

@app.get("/chatbot")
def chatbot(query: str = Query(..., description="Câu hỏi về nhà hàng hoặc ẩm thực")):
    """Chatbot tư vấn nhà hàng và đưa ra gợi ý khác nếu không phù hợp."""
    if not query:
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống.")

    # Lọc các nhà hàng có thông tin đầy đủ
    filtered_df = df.dropna(subset=['name', 'address', 'cuisines', 'average_score', 'priceRange', 'timeOpen'])

    # Tạo danh sách nhà hàng
    restaurant_list = [
        {
            "name": row['name'],
            "address": row['address'],
            "cuisines": row['cuisines'],
            "rating": row['average_score'],
            "price_range": row['priceRange'],
            "opening_hours": row['timeOpen']
        }
        for _, row in filtered_df.iterrows()
    ]

    # Tạo lời nhắc cho chatbot
    prompt = f"""
    Bạn là một chatbot tư vấn nhà hàng cho người dùng trên website.
    Người dùng hỏi: '{query}'
    
    Hãy trả lời câu hỏi của người dùng một cách rõ ràng và dễ hiểu.
    Nếu không tìm thấy nhà hàng phù hợp, hãy đưa ra các gợi ý về món ăn hoặc hoạt động liên quan đến ẩm thực.

    Dưới đây là danh sách nhà hàng hiện có:
    {', '.join([row['name'] for _, row in filtered_df.iterrows()])}

    Nếu không tìm thấy nhà hàng phù hợp, hãy đề xuất các ý tưởng hoặc món ăn mà người dùng có thể thích, dựa trên câu hỏi của họ.
    """

    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(prompt)
    
    return {
        "query": query,
        "response": response.text,
        "restaurant_suggestions": restaurant_list
    }



# Chạy FastAPI trên VS Code
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)