import os
import google.generativeai as genai
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import pandas as pd
from geopy.distance import geodesic
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
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

# Đọc dữ liệu từ file CSV
FILE_PATH = "item-meta-updated.csv"
try:
    df = pd.read_csv(FILE_PATH)
    df = df[['_id', 'image', 'name','address', 'cuisines', 'average_score', 'priceRange', 'timeOpen', 'latitude', 'longitude']].dropna()
except FileNotFoundError:
    raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {FILE_PATH}")
except KeyError as e:
    raise KeyError(f"Các cột bị thiếu trong CSV: {e}")

# Hàm tính khoảng cách giữa hai điểm (toạ độ)
def calculate_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).kilometers

# API chatbot
@app.get("/chatbot")
def chatbot(
    query: str = Query(..., description="Câu hỏi về nhà hàng hoặc ẩm thực"),
    user_lat: Optional[float] = None,  # Toạ độ người dùng (tùy chọn)
    user_lon: Optional[float] = None   # Toạ độ người dùng (tùy chọn)
):
    """Chatbot tư vấn nhà hàng và đưa ra gợi ý khác nếu không phù hợp."""
    if not query:
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống.")

    # Lọc các nhà hàng có thông tin đầy đủ
    print(df.columns.tolist())  # In ra danh sách tên cột có trong DataFrame


    filtered_df = df.dropna(subset=['_id', 'image' ,'name', 'address', 'cuisines', 'average_score', 'priceRange', 'timeOpen'])

    # Tạo danh sách nhà hàng
    restaurant_list = [
        {   
            "_id": row['_id'],
            "image": row['image'],
            "name": row['name'],
            "address": row['address'],
            "cuisines": row['cuisines'],
            "rating": row['average_score'],
            "price_range": row['priceRange'],
            "opening_hours": row['timeOpen'],
        }
        for _, row in filtered_df.iterrows()
    ]
    
    # Nếu có toạ độ người dùng, tính khoảng cách và gợi ý các nhà hàng gần nhất
    if user_lat is not None and user_lon is not None:
        # Tính khoảng cách giữa nhà hàng và người dùng
        for restaurant in restaurant_list:
            restaurant["distance"] = calculate_distance(user_lat, user_lon, restaurant["latitude"], restaurant["longitude"])
        
        # Sắp xếp danh sách nhà hàng theo khoảng cách (từ gần đến xa)
        restaurant_list = sorted(restaurant_list, key=lambda x: x["distance"])

    # Trả về danh sách nhà hàng mà không có tọa độ
    return {
        "query": query,
        "response": "Dưới đây là các nhà hàng mà bạn có thể tham khảo.",
        "restaurant_suggestions": restaurant_list
    }

# Các hàm gợi ý nhà hàng (dựa trên tương tác và nội dung)
def load_data():
    interactions_path = 'interactions.csv'  # Đặt lại đường dẫn chính xác
    item_meta_path = 'item-meta-updated.csv'  # Đặt lại đường dẫn chính xác
    interactions_df = pd.read_csv(interactions_path)
    item_meta_df = pd.read_csv(item_meta_path)
    return interactions_df, item_meta_df

def preprocess_data(interactions_df, item_meta_df):
    interactions_df.dropna(subset=['USER_ID', 'ITEM_ID'], inplace=True)
    item_meta_df.dropna(subset=['_id', 'name', 'cuisines', 'average_score', 'latitude', 'longitude'], inplace=True)

    vectorizer = TfidfVectorizer()
    cuisine_features = vectorizer.fit_transform(item_meta_df['cuisines'].astype(str))

    scaler = MinMaxScaler()
    item_meta_df['average_score_scaled'] = scaler.fit_transform(item_meta_df[['average_score']])

    return cuisine_features, item_meta_df

def generate_content_features(cuisine_features, item_meta_df):
    content_features = pd.DataFrame(cuisine_features.toarray(), index=item_meta_df['_id'])
    content_features['average_score'] = item_meta_df['average_score_scaled'].values
    return content_features

def calculate_content_similarity(content_features):
    content_similarity = cosine_similarity(content_features)
    content_similarity_df = pd.DataFrame(content_similarity, index=content_features.index, columns=content_features.index)
    return content_similarity_df

def create_interaction_matrix(interactions_df):
    interaction_matrix = interactions_df.pivot_table(
        index='USER_ID', columns='ITEM_ID', values='EVENT_TYPE', aggfunc='count', fill_value=0
    )
    return interaction_matrix

def calculate_interaction_similarity(interaction_matrix):
    interaction_similarity = cosine_similarity(interaction_matrix.T)
    interaction_similarity_df = pd.DataFrame(interaction_similarity, index=interaction_matrix.columns, columns=interaction_matrix.columns)
    return interaction_similarity_df

def get_top_similar_restaurants(restaurant_id: str, content_similarity_df, interaction_similarity_df, item_meta_df, top_n: int = 5, user_lat=None, user_lon=None) -> List[str]:
    if restaurant_id not in interaction_similarity_df.index or restaurant_id not in content_similarity_df.index:
        return []
    
    interaction_similar = interaction_similarity_df[restaurant_id]
    content_similar = content_similarity_df[restaurant_id]
    
    combined_similarity = (interaction_similar + content_similar) / 2  # Kết hợp hai nguồn
    
    if user_lat is not None and user_lon is not None:
        distances = item_meta_df.set_index('_id').apply(
            lambda row: calculate_distance(user_lat, user_lon, row['latitude'], row['longitude']), axis=1
        )
        distance_weight = 1 / (1 + distances)  # Giảm trọng số theo khoảng cách
        combined_similarity *= distance_weight  # Áp dụng trọng số khoảng cách

    similar_restaurants = combined_similarity.sort_values(ascending=False).iloc[1:top_n+1]
    return similar_restaurants.index.tolist()

def recommend_for_user(user_id: str, current_restaurant_id: str, interaction_matrix, content_similarity_df, interaction_similarity_df, item_meta_df, top_n: int = 5, user_lat=None, user_lon=None) -> List[str]:
    # Kiểm tra nếu người dùng không có trong interaction matrix, thì dựa vào sự tương đồng giữa các nhà hàng
    if user_id not in interaction_matrix.index:
        # Nếu không có người dùng trong dữ liệu, sử dụng sự tương tự giữa các nhà hàng
        recommended_restaurants = get_top_similar_restaurants(current_restaurant_id, content_similarity_df, interaction_similarity_df, item_meta_df, top_n, user_lat, user_lon)
        return recommended_restaurants
    
    # Nếu người dùng có trong dữ liệu, tiếp tục dựa vào thông tin tương tác của người dùng
    user_items = interaction_matrix.loc[user_id]
    interacted_restaurants = user_items[user_items > 0].index.tolist()
    recommended_restaurants = set()

    # Thêm nhà hàng tương tự từ nhà hàng hiện tại
    recommended_restaurants.update(get_top_similar_restaurants(current_restaurant_id, content_similarity_df, interaction_similarity_df, item_meta_df, top_n, user_lat, user_lon))

    # Thêm nhà hàng tương tự từ các nhà hàng mà người dùng đã tương tác
    for restaurant in interacted_restaurants:
        recommended_restaurants.update(get_top_similar_restaurants(restaurant, content_similarity_df, interaction_similarity_df, item_meta_df, top_n, user_lat, user_lon))

    return list(recommended_restaurants)[:top_n]


# Lấy thông tin chi tiết các nhà hàng
def get_restaurant_details(restaurant_ids: List[str], item_meta_df):
    """Lấy thông tin chi tiết của các nhà hàng từ danh sách ID."""
    return item_meta_df[item_meta_df['_id'].isin(restaurant_ids)][['_id','image', 'name', 'cuisines', 'address', 'average_score', 'priceRange', 'timeOpen', 'latitude', 'longitude']]


# API Endpoint để gợi ý nhà hàng cho người dùng
@app.get("/recommendations")
async def recommend(
    user_id: str,
    current_restaurant_id: str,
    top_n: int = 5,
    user_lat: Optional[float] = None,
    user_lon: Optional[float] = None
):
    interactions_df, item_meta_df = load_data()
    cuisine_features, item_meta_df = preprocess_data(interactions_df, item_meta_df)
    content_features = generate_content_features(cuisine_features, item_meta_df)
    content_similarity_df = calculate_content_similarity(content_features)
    interaction_matrix = create_interaction_matrix(interactions_df)
    interaction_similarity_df = calculate_interaction_similarity(interaction_matrix)
    
    recommended_restaurant_ids = recommend_for_user(user_id, current_restaurant_id, interaction_matrix, content_similarity_df, interaction_similarity_df, item_meta_df, top_n, user_lat, user_lon)
    recommended_restaurants = get_restaurant_details(recommended_restaurant_ids, item_meta_df)
    
    return {"recommended_restaurants": recommended_restaurants.to_dict(orient="records")}
