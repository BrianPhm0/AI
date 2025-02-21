

Vào terminal của Dự án, chạy lệnh uvicorn chatbotandrecommend:app --host 127.0.0.1 --port 8000 --reload




Khi call recommendations, nên cung cấp tung độ, hoành độ cho user



//Sau đó dùng node js để call api

npm install axios

const axios = require('axios');

// Gửi yêu cầu tới API FastAPI
async function getChatbotResponse(query, userLat, userLon) {
  try {
    const response = await axios.get('http://127.0.0.1:8000/chatbot', {
      params: {
        query: query,
        user_lat: userLat,
        user_lon: userLon
      }
    });

    console.log("Chatbot response:", response.data);
    return response.data;
  } catch (error) {
    console.error("Error calling chatbot API:", error);
    throw new Error("Không thể nhận dữ liệu từ chatbot.");
  }
}

// Ví dụ sử dụng
getChatbotResponse('Tìm nhà hàng Ý', 21.0285, 105.8542);


const axios = require('axios');

// Gửi yêu cầu tới API FastAPI để nhận gợi ý nhà hàng

Khi call recommendations, nên cung cấp tung độ, hoành độ cho user

async function getRecommendations(userId, currentRestaurantId, topN, userLat, userLon) {
  try {
    const response = await axios.get('http://127.0.0.1:8000/recommendations', {
      params: {
        user_id: userId,
        current_restaurant_id: currentRestaurantId,
        top_n: topN,
        user_lat: userLat,
        user_lon: userLon
      }
    });

    console.log("Recommended restaurants:", response.data);
    return response.data;
  } catch (error) {
    console.error("Error calling recommendations API:", error);
    throw new Error("Không thể nhận dữ liệu từ API gợi ý nhà hàng.");
  }
}

// Ví dụ sử dụng
getRecommendations('user123', 'restaurant456', 5, 21.0285, 105.8542);
