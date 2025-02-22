import requests
from datetime import datetime
from typing import Dict, List, Any
import os
import base64
from openai import OpenAI
import statistics
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Your existing StockNewsProcessor and StockNewsAnalyzer classes remain unchanged
class StockNewsProcessor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.source_reliability = {
            'Reuters': 5,
            'Bloomberg': 5,
            'CNBC': 4,
            'Financial Times': 5,
            'Wall Street Journal': 5,
            'Benzinga': 3,
            'Motley Fool': 3,
            'Zacks': 3
        }
    
    def fetch_sentiment_data(self, ticker: str) -> Dict[str, Any]:
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': ticker,
            'apikey': self.api_key
        }
        response = requests.get(self.base_url, params=params)
        return response.json()
    
    def process_news_items(self, data: Dict[str, Any], ticker: str, min_relevance: float = 0.5) -> List[Dict[str, Any]]:
        processed_items = []
        for item in data.get('feed', []):
            ticker_data = None
            for tick in item.get('ticker_sentiment', []):
                if tick['ticker'] == ticker.upper():
                    ticker_data = tick
                    break
            if not ticker_data or float(ticker_data.get('relevance_score', 0)) < min_relevance:
                continue
            time_str = item.get('time_published', '')
            try:
                published_date = datetime.strptime(time_str, '%Y%m%d%H%M%S')
                formatted_date = published_date.strftime('%Y-%m-%d %H:%M:%S')
            except ValueError:
                formatted_date = time_str
            source = item.get('source')
            reliability_rating = self.source_reliability.get(source, 2)
            processed_item = {
                'title': item.get('title'),
                'published_date': formatted_date,
                'summary': item.get('summary'),
                'source': source,
                'source_reliability': reliability_rating,
                'url': item.get('url'),
                'overall_sentiment': item.get('overall_sentiment_score'),
                'sentiment_label': item.get('overall_sentiment_label'),
                'relevance_score': ticker_data.get('relevance_score'),
                'ticker_specific_sentiment': ticker_data.get('ticker_sentiment_score'),
                'ticker_sentiment_label': ticker_data.get('ticker_sentiment_label')
            }
            processed_items.append(processed_item)
        return sorted(processed_items, 
                      key=lambda x: (float(x['relevance_score']), x['source_reliability']), 
                      reverse=True)
    
    def get_sentiment_summary(self, ticker: str, min_relevance: float = 0.5) -> Dict[str, Any]:
        data = self.fetch_sentiment_data(ticker)
        return {
            'sentiment_definition': data.get('sentiment_score_definition'),
            'relevance_score_definition': data.get('relevance_score_definition'),
            'news_items': self.process_news_items(data, ticker, min_relevance)
        }

class StockNewsAnalyzer(StockNewsProcessor):
    def __init__(self, alpha_vantage_key: str, openai_key: str, openai_org_id: str):
        super().__init__(alpha_vantage_key)
        self.openai_client = OpenAI(
            api_key=openai_key,
            organization=openai_org_id
        )
    
    def format_news_for_analysis(self, news_items: List[Dict[str, Any]]) -> str:
        formatted_text = "Recent News Items and Sentiment Analysis:\n\n"
        for idx, item in enumerate(news_items, 1):
            formatted_text += f"{idx}. {item['title']}\n"
            formatted_text += f"Published: {item['published_date']}\n"
            formatted_text += f"Source: {item['source']} (Reliability: {item['source_reliability']}/5)\n"
            formatted_text += f"Relevance Score: {float(item['relevance_score']):.2f}\n"
            formatted_text += f"Ticker-Specific Sentiment: {item['ticker_sentiment_label']} "
            formatted_text += f"(Score: {float(item['ticker_specific_sentiment']):.3f})\n"
            formatted_text += f"Summary: {item['summary'][:200]}...\n\n"
        return formatted_text

    def analyze_news_with_ai(self, news_data: str) -> str:
        try:
            prompt = f"""Please analyze these financial news items and provide:
1. Overall market sentiment analysis
2. Key patterns or themes in the coverage
3. Notable changes in sentiment over time
4. Potential market implications
5. Important factors for investors to watch

News Data:
{news_data}"""
            response = self.openai_client.chat.completions.create(
                model="o1-mini",
                messages=[
                    {"role": "user", "content": "Provide a detailed analysis of this stock using the following data"
                     + prompt + "Do not just take sentiment ratings at face value; be sure to analyze the summaries of the articles provided."
                     + "Also, in the end, rate this stock's short term trends from 0 to 100, and also rate the underlying fundamentals from 0 to 100."}
                ],
                temperature=1,
                max_completion_tokens=3000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating AI analysis: {str(e)}"

    def get_complete_analysis(self, ticker: str) -> Dict[str, Any]:
        sentiment_data = self.get_sentiment_summary(ticker)
        formatted_news = self.format_news_for_analysis(sentiment_data['news_items'])
        ai_analysis = self.analyze_news_with_ai(formatted_news)
        return {
            'raw_sentiment_data': sentiment_data,
            'ai_analysis': ai_analysis
        }

# Load environment variables
alpha_vantage_key = os.environ.get('ALPHA_VANTAGE_KEY')
openai_key = os.environ.get('OPENAI_KEY')
openai_org_id = os.environ.get('OPENAI_ORG_ID')

# Validate environment variables
if not all([alpha_vantage_key, openai_key, openai_org_id]):
    raise ValueError("Missing required environment variables: ALPHA_VANTAGE_KEY, OPENAI_KEY, or OPENAI_ORG_ID")

# Initialize analyzer globally
analyzer = StockNewsAnalyzer(
    alpha_vantage_key=alpha_vantage_key,
    openai_key=openai_key,
    openai_org_id=openai_org_id
)

@app.route('/')
def serve_interface():
    try:
        with open('ai.html', 'r') as file:
            return file.read()
    except FileNotFoundError:
        return "Error: ai.html not found", 404

@app.route('/process', methods=['POST'])
def process_input():
    try:
        data = request.get_json()
        if not data or 'input' not in data:
            return jsonify({'error': 'No ticker provided'}), 400
        ticker = data['input'].strip().upper()
        logger.info(f"Processing ticker: {ticker}")
        analysis = analyzer.get_complete_analysis(ticker)
        disclaimer = {
            "disclaimer": "IMPORTANT: This analysis is for informational purposes only and should not be considered as financial advice. The analysis is based on news sentiment and may not reflect all market factors. Past performance is not indicative of future results. Always conduct your own research and consult with a qualified financial advisor before making investment decisions."
        }
        response_data = {
            **disclaimer,
            'news_items': [
                {
                    'title': item['title'],
                    'published_date': item['published_date'],
                    'source': f"{item['source']} (Reliability: {item['source_reliability']}/5)",
                    'relevance_score': f"{float(item['relevance_score']):.2f}",
                    'sentiment': f"{item['ticker_sentiment_label']} (Score: {float(item['ticker_specific_sentiment']):.3f})",
                    'summary': item['summary'][:200] + "..."
                }
                for item in analysis['raw_sentiment_data']['news_items']
            ],
            'ai_analysis': analysis['ai_analysis']
        }
        print("Sending response:", response_data)
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error processing ticker: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Stock Analysis Server Starting...")
    # Test API connectivity (for local development only)
    try:
        test_response = requests.get(
            "https://www.alphavantage.co/query",
            params={"function": "TIME_SERIES_INTRADAY", "symbol": "IBM", "interval": "1min", "apikey": analyzer.api_key}
        )
        test_response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error: Could not connect to Alpha Vantage API: {e}")
        exit(1)
    host = '127.0.0.1'
    port = 5000
    print(f"API connections verified successfully")
    print(f"Access the interface at http://{host}:{port}")
    app.run(host=host, port=port, debug=True)
