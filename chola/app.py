import os
import random
import json
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file using absolute path
from pathlib import Path
env_path = Path(__file__).parent / "environment.env"
load_dotenv(env_path)

# --- Initialize Flask App ---
app = Flask(__name__, template_folder='templates')
CORS(app) # Enable Cross-Origin Resource Sharing

# --- Configure Gemini API ---
# Note: It's highly recommended to set your API key in the .env file
# for security.
try:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
    else:
        print("⚠️ WARNING: GEMINI_API_KEY not found in .env file. Chatbot will not function.")
except Exception as e:
    print(f"⚠️ ERROR: Could not configure Gemini API: {e}")


# --- Helper Functions (Simulating Models) ---
# In a real-world scenario, these functions would contain complex logic,
# machine learning models, and database lookups.

def get_price_prediction(commodity, market, days_ahead):
    """Placeholder for a real ML price prediction model."""
    base_prices = {
        'wheat': 2200, 'rice': 3000, 'cotton': 6000, 'onion': 1700,
        'potato': 1500, 'maize': 1800, 'tomato': 1200
    }
    price = base_prices.get(commodity.lower(), 2000 + random.uniform(0, 5000))
    price += (days_ahead * (random.uniform(-5, 10)))
    price *= 1.05 if market.lower() in ['mumbai', 'delhi'] else 1.01
    return round(price, 2)

def get_sentiment_analysis(text):
    """Placeholder for a real NLP sentiment analysis model."""
    score = random.uniform(-1, 1)
    if score > 0.3:
        sentiment = "Positive"
    elif score < -0.3:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return {'sentiment': sentiment, 'score': score}

def get_historical_data(predicted_price, days_ahead):
    """Generates simulated historical and future price data for charts."""
    dates, prices = [], []
    today = datetime.now()
    base = predicted_price / (1 + (random.uniform(-0.02, 0.05)))
    # Generate past 30 days of data
    for i in range(30, -1, -1):
        d = today - timedelta(days=i)
        dates.append(d.strftime('%Y-%m-%d'))
        price = base + (i**0.5 * random.uniform(-15, 20)) + random.uniform(-25, 25)
        prices.append(round(price))
    # Add the future predicted price
    future_date = today + timedelta(days=days_ahead)
    dates.append(future_date.strftime('%Y-%m-%d'))
    prices.append(predicted_price)
    return {'dates': dates, 'prices': prices}

def get_xai_insights():
    """Placeholder for Explainable AI (XAI) feature importance results."""
    factors = [
        {'factor': 'Recent rainfall', 'impact': random.choice(['Positive', 'Negative'])},
        {'factor': 'Mandi arrivals', 'impact': random.choice(['Positive', 'Negative'])},
        {'factor': 'News sentiment', 'impact': 'Positive'},
        {'factor': 'Global market trends', 'impact': 'Negative'},
        {'factor': 'Fuel prices', 'impact': 'Negative'},
        {'factor': 'Seasonal demand', 'impact': 'Positive'},
    ]
    random.shuffle(factors)
    return factors[:4]

def get_weather_data(market):
    """Placeholder for a real weather API call."""
    conditions = ["Clear", "Partly Cloudy", "Light Rain", "Sunny"]
    temp = 28 + random.randint(0, 10)
    condition = random.choice(conditions)
    impact = "No significant impact on crop prices expected."
    if condition == "Light Rain" and random.random() > 0.5:
        impact = "Recent light rain is favorable for sowing, potentially stabilizing prices."
    if temp > 35:
        impact = "High temperatures may stress crops, potentially leading to a slight price increase if sustained."
    return {'market': market, 'temp': temp, 'condition': condition, 'impact': impact}

def get_crop_recommendation(soil, rainfall, ph, temp):
    """Placeholder for a crop recommendation engine."""
    crop, reason = "Wheat", "Conditions are generally suitable for wheat cultivation."
    if soil == 'Black' and rainfall > 1200:
        crop, reason = "Cotton", "Black soil and high rainfall are ideal for cotton."
    elif soil == 'Alluvial' and rainfall > 1500:
        crop, reason = "Rice", "Alluvial soil with abundant water supply is perfect for rice paddies."
    elif temp > 28 and rainfall < 800:
        crop, reason = "Bajra", "This crop is resilient to high temperatures and lower rainfall."
    return {'crop': crop, 'reason': reason}

def get_latest_news():
    """Placeholder for a news scraping service."""
    return [
        "Government announces new MSP for Kharif crops, farmers hopeful.",
        "Monsoon forecast predicts above-average rainfall in Northern India.",
        "Global wheat prices surge due to supply chain disruptions.",
        "New pest-resistant cotton variety shows promising results in trials."
    ]


# --- API Endpoints ---

@app.route('/')
def index():
    """Serves the main dashboard HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def handle_predict():
    """Handles the main price prediction request."""
    data = request.json
    commodity = data.get('commodity', 'Wheat')
    market = data.get('market', 'Delhi')
    days_ahead = int(data.get('daysAhead', 7))

    # Get all required data from backend helpers
    predicted_price = get_price_prediction(commodity, market, days_ahead)
    historical_data = get_historical_data(predicted_price, days_ahead)
    xai_insights = get_xai_insights()

    # Consolidate into a single response
    response = {
        'prediction': {
            'predicted_modal_price_INR': predicted_price,
            'unit': "INR/Quintal"
        },
        'chartData': historical_data,
        'xai': xai_insights
    }
    return jsonify(response)

@app.route('/analyze-sentiment', methods=['POST'])
def handle_sentiment():
    """Handles sentiment analysis requests."""
    data = request.json
    text = data.get('text', '')
    result = get_sentiment_analysis(text)
    return jsonify(result)

@app.route('/weather', methods=['GET'])
def handle_weather():
    """Handles weather data requests."""
    market = request.args.get('market', 'Delhi')
    result = get_weather_data(market)
    return jsonify(result)
    
@app.route('/recommend-crop', methods=['POST'])
def handle_crop_recommendation():
    """Handles crop recommendation requests."""
    data = request.json
    result = get_crop_recommendation(
        data.get('soil'), data.get('rainfall'), data.get('ph'), data.get('temp')
    )
    return jsonify(result)

@app.route('/news', methods=['GET'])
def handle_news():
    """Handles fetching latest news."""
    news_items = get_latest_news()
    return jsonify(news_items)

@app.route('/chat', methods=['POST'])
def handle_chat():
    """Handles chatbot requests by proxying to the Gemini API."""
    user_query = request.json.get("query")
    if not gemini_api_key:
        return jsonify({"response": "API Key is not configured on the server. Please set GEMINI_API_KEY in your environment."}), 400

    try:
        # Always use the server-side API key
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"You are an agricultural expert bot named AgriBERT. Answer the following user query about Indian agriculture in a concise and helpful way: \"{user_query}\""
        chat_response = model.generate_content(prompt)
        return jsonify({"response": chat_response.text})
    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        return jsonify({"response": f"Sorry, an error occurred while connecting to the AI service: {e}"}), 500


# --- Main Execution ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)