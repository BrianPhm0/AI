import pandas as pd
from fastapi import FastAPI, Query
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from geopy.distance import geodesic
from typing import List, Optional

app = FastAPI()

# Đọc dữ liệu từ file
def load_data():
    interactions_path = '/kaggle/input/item-meta/interactions.csv'
    item_meta_path = '/kaggle/input/item-meup/item-meta-updated.csv'
    interactions_df = pd.read_csv(interactions_path)
    item_meta_df = pd.read_csv(item_meta_path)

    return interactions_df, item_meta_df

# Tiền xử lý dữ liệu
def preprocess_data(interactions_df, item_meta_df):
    interactions_df.dropna(subset=['USER_ID', 'ITEM_ID'], inplace=True)
    item_meta_df.dropna(subset=['_id', 'name', 'cuisines', 'average_score', 'latitude', 'longitude'], inplace=True)

    # Mã hóa đặc trưng nội dung của nhà hàng
    vectorizer = TfidfVectorizer()
    cuisine_features = vectorizer.fit_transform(item_meta_df['cuisines'].astype(str))

    # Chuẩn hóa điểm đánh giá trung bình
    scaler = MinMaxScaler()
    item_meta_df['average_score_scaled'] = scaler.fit_transform(item_meta_df[['average_score']])

    return cuisine_features, item_meta_df

# Kết hợp đặc trưng nội dung
def generate_content_features(cuisine_features, item_meta_df):
    content_features = pd.DataFrame(cuisine_features.toarray(), index=item_meta_df['_id'])
    content_features['average_score'] = item_meta_df['average_score_scaled'].values
    return content_features

# Tính toán ma trận độ tương đồng giữa các nhà hàng dựa trên nội dung
def calculate_content_similarity(content_features):
    content_similarity = cosine_similarity(content_features)
    content_similarity_df = pd.DataFrame(content_similarity, index=content_features.index, columns=content_features.index)
    return content_similarity_df

# Tạo ma trận tương tác giữa người dùng và nhà hàng
def create_interaction_matrix(interactions_df):
    interaction_matrix = interactions_df.pivot_table(
        index='USER_ID', columns='ITEM_ID', values='EVENT_TYPE', aggfunc='count', fill_value=0
    )
    return interaction_matrix

# Tính toán ma trận độ tương đồng giữa các nhà hàng dựa trên tương tác
def calculate_interaction_similarity(interaction_matrix):
    interaction_similarity = cosine_similarity(interaction_matrix.T)
    interaction_similarity_df = pd.DataFrame(interaction_similarity, index=interaction_matrix.columns, columns=interaction_matrix.columns)
    return interaction_similarity_df

# Tính khoảng cách địa lý giữa hai địa điểm
def calculate_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).kilometers

# Gợi ý nhà hàng dựa trên độ tương đồng và khoảng cách địa lý
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

# Gợi ý nhà hàng cho người dùng dựa trên lịch sử tương tác và khoảng cách
def recommend_for_user(user_id: str, current_restaurant_id: str, interaction_matrix, content_similarity_df, interaction_similarity_df, item_meta_df, top_n: int = 5, user_lat=None, user_lon=None) -> List[str]:
    if user_id not in interaction_matrix.index:
        return []
    
    user_items = interaction_matrix.loc[user_id]
    interacted_restaurants = user_items[user_items > 0].index.tolist()
    recommended_restaurants = set()
    
    # Gợi ý dựa trên nhà hàng hiện tại
    recommended_restaurants.update(get_top_similar_restaurants(current_restaurant_id, content_similarity_df, interaction_similarity_df, item_meta_df, top_n, user_lat, user_lon))
    
    # Gợi ý dựa trên lịch sử tương tác
    for restaurant in interacted_restaurants:
        recommended_restaurants.update(get_top_similar_restaurants(restaurant, content_similarity_df, interaction_similarity_df, item_meta_df, top_n, user_lat, user_lon))
    
    return list(recommended_restaurants)[:top_n]

# Lấy thông tin chi tiết các nhà hàng
def get_restaurant_details(restaurant_ids: List[str], item_meta_df):
    return item_meta_df[item_meta_df['_id'].isin(restaurant_ids)][['_id', 'name', 'cuisines', 'address', 'average_score', 'latitude', 'longitude']]

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
    
    recommended_restaurants = recommend_for_user(
        user_id, 
        current_restaurant_id, 
        interaction_matrix, 
        content_similarity_df, 
        interaction_similarity_df, 
        item_meta_df, 
        top_n=top_n, 
        user_lat=user_lat, 
        user_lon=user_lon
    )

    recommended_restaurant_details = get_restaurant_details(recommended_restaurants, item_meta_df)
    
    return recommended_restaurant_details.to_dict(orient="records")

# Chạy ứng dụng FastAPI với Uvicorn
# uvicorn script_name:app --reload
