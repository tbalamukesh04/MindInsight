from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import math
import json
import logging
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from peft import PeftModel, PeftConfig
from loguru import logger
from marshmallow import ValidationError
from schemas.response_schemas import (
    ClassificationResultSchema,
    BatchClassificationResponseSchema,
    CumulativeResultSchema
)

# Configure logging
logger.add("app.log", rotation="10 MB")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure upload directory
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models')
BASE_MODEL = 'FacebookAI/roberta-base'

class QuestionnaireClassifier:
    def __init__(self, base_model_name, adapter_path):
        try:
            # Load base model
            self.tokenizer = RobertaTokenizer.from_pretrained(base_model_name)
            base_model = RobertaForSequenceClassification.from_pretrained(
                base_model_name,
                num_labels=2
            )
            
            # Load adapter with correct config
            self.model = PeftModel.from_pretrained(
                base_model,
                adapter_path,
                config=PeftConfig.from_pretrained(adapter_path))
            
            self.model = self.model.merge_and_unload()
            self.model.eval()
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Loading error: {str(e)}")
            raise
    
    def process_batch(self, texts):
        """Process a batch of texts and return classification results"""
        try:
            results = []
            for text in texts:
                if not text:
                    results.append({
                        'predicted_class': 0,
                        'confidence': 1.0
                    })
                    continue
                
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                pred_class = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][pred_class].item()
                
                results.append({
                    'predicted_class': pred_class,
                    'confidence': confidence
                })
            
            return results
        
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            raise

    def process_text(self, text):
        """Classify a single text"""
        return self.process_batch([text])[0]


# Initialize classifier and schemas
text_processor = QuestionnaireClassifier(BASE_MODEL, MODEL_PATH)
cumulative_schema = CumulativeResultSchema()
classification_schema = ClassificationResultSchema()

# -------------------- Routes --------------------

@app.route('/classify-questionnaire', methods=['POST'])
def classify_questionnaire():
    """Endpoint for questionnaire-style classification (12 inputs)"""
    try:
        texts = [request.form.get(f'text{i}', '').strip() for i in range(1, 13)]
        logger.info(f"Received questionnaire texts: {texts}")
        
        if not any(texts):
            logger.warning("All text inputs empty")
            return jsonify({"error": "At least one text input required"}), 400

        individual_results = text_processor.process_batch(texts)

        # Debug logging
        logger.info(f"Processing results: {json.dumps(individual_results, indent=2)}")

        if not individual_results or len(individual_results) != 12:
            logger.error("Invalid processing results")
            return jsonify({"error": "Analysis failed"}), 500

        # Clinical weights (Q1-Q12)
        weights = [0.15, 0.12, 0.15, 0.10, 0.08, 
                   0.10, 0.10, 0.05, 0.20, 0.02, 
                   0.02, 0.01]
        
        depression_score = 0.0
        max_possible_score = sum(weights)
        
        for idx, result in enumerate(individual_results):
            weight = weights[idx]
            class_1_prob = result['confidence'] if result['predicted_class'] == 1 else (1 - result['confidence'])
            depression_score += class_1_prob * weight
        
        # Normalize and convert to percentage
        depression_level = (depression_score / max_possible_score) * 100
        depression_level = max(55.0, min(95.0, depression_level))
        
        # Determine severity class
        final_class = 1 if depression_level >= 70 else 0
        
        response = {
            'final_class': final_class,
            'depression_level': depression_level,
            'individual_results': individual_results
        }
        
        return jsonify(cumulative_schema.dump(response)), 200
    
    except Exception as e:
        logger.error(f"Questionnaire error: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/classify', methods=['POST'])
def classify_text():
    """Endpoint for single text classification"""
    try:
        text = request.json.get('text') if request.is_json else request.form.get('text')
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        result = text_processor.process_text(text)
        return jsonify(classification_schema.dump(result)), 200
    
    except ValidationError as err:
        logger.error(f"Validation error: {err}")
        return jsonify({"error": err.messages}), 400
    except Exception as e:
        logger.error(f"Classification error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/questionnaire')
def questionnaire():
    return render_template('questionnaire.html')

@app.route('/results')
def results():
    return render_template('results.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)