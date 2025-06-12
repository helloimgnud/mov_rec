from flask import Flask, render_template, request, jsonify
import logging
import traceback
from model_in_use import AutoRec, UserFeaturesNet
from recommender import get_recommendations
from utils.process_data import map_age, map_occupation
import json
app = Flask(__name__)


logging.basicConfig(level=logging.ERROR, filename='app.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:

        data = request.json
        age = float(map_age(int(data['age'])))
        gender = float(data['gender'])
        occupation = float(map_occupation(float(data['occupation']))) 

        recommended_ids = get_recommendations(age, gender, occupation)
        return jsonify(recommended_ids)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
        logging.error("Exception occurred", exc_info=True)


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "API is running"})


if __name__ == '__main__':
    app.run(debug=True)
