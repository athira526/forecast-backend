from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, jwt_required, get_jwt_identity
import pandas as pd
import os
from datetime import datetime
import torch
import numpy as np
import logging

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'default-secret-for-local-testing')
jwt = JWTManager(app)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_FOLDER = 'Uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory storage for user predictions
user_predictions = {}

# Mock user store mapping (replace with database later)
USER_STORE_MAPPING = {
    'user1@example.com': 'User1 Store',
    'user2@example.com': 'User2 Store',
    # Add more users as needed
}

# Load the scripted TFT model
try:
    model = torch.jit.load("tft_traced_model.pt")
    model.eval()
    logger.info("✅ TorchScript model loaded successfully.")
except Exception as e:
    logger.error("❌ Failed to load model: %s", str(e))
    raise e

def validate_excel_data(df):
    required_columns = ['date', 'history', 'onpromotion', 'is_holiday', 'transactions', 'store_nbr', 'item_nbr']
    return all(col in df.columns for col in required_columns)

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not Found", "message": "The requested endpoint was not found on the server.", "status": 404}), 404

@app.route('/user', methods=['GET'])
@jwt_required()
def get_user():
    try:
        user_email = get_jwt_identity()
        logger.info("Fetching store name for user: %s", user_email)
        store_name = USER_STORE_MAPPING.get(user_email, 'Default Store')
        return jsonify({
            'username': user_email,
            'store_name': store_name
        }), 200
    except Exception as e:
        logger.error("Error fetching user data: %s", str(e))
        return jsonify({"error": "Failed to fetch user data"}), 500

@app.route('/upload', methods=['POST'])
@jwt_required()
def upload_file():
    user_email = get_jwt_identity()
    logger.info("Received /upload request from user: %s", user_email)
    if 'file' not in request.files:
        logger.error("No file provided")
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        logger.error("No file selected")
        return jsonify({"error": "No file selected"}), 400
    
    if file and file.filename.endswith('.xlsx'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"upload_{user_email}_{timestamp}.xlsx"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        logger.info("Saving file to %s", file_path)
        file.save(file_path)
        
        try:
            df = pd.read_excel(file_path)
            if not validate_excel_data(df):
                logger.error("Invalid Excel format")
                return jsonify({"error": "Invalid Excel format"}), 400
            
            row_count = len(df)
            logger.info("File processed successfully, rows: %d", row_count)
            return jsonify({
                "message": "File uploaded and validated successfully",
                "filename": filename,
                "row_count": row_count
            }), 200
        except Exception as e:
            logger.error("Error processing file: %s", str(e))
            return jsonify({"error": str(e)}), 500
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info("File %s deleted", file_path)
    else:
        logger.error("Invalid file format")
        return jsonify({"error": "Invalid file format, only .xlsx allowed"}), 400

@app.route('/forecast', methods=['POST'])
@jwt_required()
def forecast():
    try:
        user_email = get_jwt_identity()
        logger.info("Received /forecast request from user: %s", user_email)
        data = request.get_json()
        forecast_days = min(data.get('forecast_days', 7), 30)
        custom_is_holiday = data.get('is_holiday', None)
        custom_onpromotion = data.get('onpromotion', None)
        store_name = data.get('store_name', 'Store 1')
        item_name = data.get('item_name', 'Item 1')

        upload_dir = UPLOAD_FOLDER
        user_files = [f for f in os.listdir(upload_dir) if f.startswith(f"upload_{user_email}_") and f.endswith('.xlsx')]
        if not user_files:
            logger.error("No uploaded files found for user: %s", user_email)
            return jsonify({"error": "No uploaded files found for this user"}), 404
        
        latest_file = max(
            [os.path.join(upload_dir, f) for f in user_files],
            key=os.path.getctime
        )
        
        df = pd.read_excel(latest_file)
        if not validate_excel_data(df):
            logger.error("Invalid data in uploaded file")
            return jsonify({"error": "Invalid data in uploaded file"}), 400
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        df['store_nbr'] = df['store_nbr'].fillna(1).astype(int)
        df['item_nbr'] = df['item_nbr'].fillna(1).astype(int)
        df['onpromotion'] = df['onpromotion'].fillna(0).astype(int)
        df['is_holiday'] = df['is_holiday'].fillna(0).astype(int)
        df['transactions'] = df['transactions'].fillna(0).astype(float)
        df['history'] = df['history'].fillna(0).astype(float)
        
        store_nbr = df['store_nbr'].iloc[-1]
        item_nbr = df['item_nbr'].iloc[-1]
        history = df['history'].tail(30).tolist()
        transactions = df['transactions'].tail(37).tolist()
        
        is_holiday = custom_is_holiday if custom_is_holiday and len(custom_is_holiday) == 37 else df['is_holiday'].tail(37).tolist()
        onpromotion = custom_onpromotion if custom_onpromotion and len(custom_onpromotion) == 37 else df['onpromotion'].tail(37).tolist()

        if len(history) < 30:
            logger.error("Insufficient history data: need 30 days")
            return jsonify({"error": "Insufficient history data: need 30 days"}), 400
        history = history[-30:]
        
        max_horizon = 30
        time_steps = 37 + max_horizon
        transactions = transactions[-37:] + [transactions[-1]] * max_horizon
        is_holiday = is_holiday[-37:] + [is_holiday[-1]] * max_horizon
        onpromotion = onpromotion[-37:] + [onpromotion[-1]] * max_horizon

        encoder_len = 30
        decoder_len = 1

        predictions = {"p10": [], "p50": [], "p90": []}
        current_history = history.copy()
        current_onpromotion = onpromotion.copy()
        current_is_holiday = is_holiday.copy()
        current_transactions = transactions.copy()

        for day in range(forecast_days):
            encoder_lengths = torch.tensor([encoder_len], dtype=torch.long)
            decoder_lengths = torch.tensor([decoder_len], dtype=torch.long)

            history_tensor = torch.tensor(current_history[-encoder_len:], dtype=torch.float).unsqueeze(0)
            transactions_tensor = torch.tensor(current_transactions[day:37+day], dtype=torch.float).unsqueeze(0)
            time_idx = torch.arange(day, 37+day, dtype=torch.float).unsqueeze(0)
            day_of_week = torch.tensor([(i % 7) for i in range(day, 37+day)], dtype=torch.float).unsqueeze(0)
            month = torch.tensor([((i % 12) + 1) for i in range(day, 37+day)], dtype=torch.float).unsqueeze(0)
            dummy1 = torch.zeros(1, 37, dtype=torch.float)
            dummy2 = torch.zeros(1, 37, dtype=torch.float)
            dummy3 = torch.zeros(1, 37, dtype=torch.float)
            dummy4 = torch.zeros(1, 37, dtype=torch.float)
            dummy5 = torch.zeros(1, 37, dtype=torch.float)

            encoder_cont = torch.stack([
                history_tensor[:, :encoder_len],
                transactions_tensor[:, :encoder_len],
                time_idx[:, :encoder_len],
                day_of_week[:, :encoder_len],
                month[:, :encoder_len],
                dummy1[:, :encoder_len],
                dummy2[:, :encoder_len],
                dummy3[:, :encoder_len],
                dummy4[:, :encoder_len],
                dummy5[:, :encoder_len]
            ], dim=-1)

            decoder_cont = torch.stack([
                history_tensor[:, -1:],
                transactions_tensor[:, encoder_len:encoder_len+decoder_len],
                time_idx[:, encoder_len:encoder_len+decoder_len],
                day_of_week[:, encoder_len:encoder_len+decoder_len],
                month[:, encoder_len:encoder_len+decoder_len],
                dummy1[:, encoder_len:encoder_len+decoder_len],
                dummy2[:, encoder_len:encoder_len+decoder_len],
                dummy3[:, encoder_len:encoder_len+decoder_len],
                dummy4[:, encoder_len:encoder_len+decoder_len],
                dummy5[:, encoder_len:encoder_len+decoder_len]
            ], dim=-1)

            store_tensor = torch.tensor([[store_nbr] * 37], dtype=torch.long)
            item_tensor = torch.tensor([[item_nbr] * 37], dtype=torch.long)
            promo_tensor = torch.tensor(current_onpromotion[day:37+day], dtype=torch.long).unsqueeze(0)
            holiday_tensor = torch.tensor(current_is_holiday[day:37+day], dtype=torch.long).unsqueeze(0)

            encoder_cat = torch.stack([
                store_tensor[:, :encoder_len],
                item_tensor[:, :encoder_len],
                promo_tensor[:, :encoder_len],
                holiday_tensor[:, :encoder_len]
            ], dim=-1)

            decoder_cat = torch.stack([
                store_tensor[:, encoder_len:encoder_len+decoder_len],
                item_tensor[:, encoder_len:encoder_len+decoder_len],
                promo_tensor[:, encoder_len:encoder_len+decoder_len],
                holiday_tensor[:, encoder_len:encoder_len+decoder_len]
            ], dim=-1)

            history_array = torch.tensor(current_history, dtype=torch.float)
            target_mean = history_array.mean()
            target_std = history_array.std() if history_array.std() > 0 else torch.tensor(1.0)
            target_scale = torch.tensor([[target_mean, target_std]], dtype=torch.float)

            model_input = {
                "encoder_cont": encoder_cont,
                "decoder_cont": decoder_cont,
                "encoder_cat": encoder_cat,
                "decoder_cat": decoder_cat,
                "encoder_lengths": encoder_lengths,
                "decoder_lengths": decoder_lengths,
                "target_scale": target_scale
            }

            with torch.no_grad():
                output = model(model_input)

            p10 = output[0][:, :, 1].squeeze().item()
            p50 = output[0][:, :, 3].squeeze().item()
            p90 = output[0][:, :, 5].squeeze().item()
            predictions["p10"].append(p10)
            predictions["p50"].append(p50)
            predictions["p90"].append(p90)

            current_history.append(p50)
            current_history = current_history[1:]

        avg_forecast = np.mean(predictions["p50"])
        suggestions = [{
            "type": "stock_adjustment",
            "message": f"Prepare stock for ~{round(avg_forecast)} units/day of {item_name} at {store_name}.",
            "confidence": 0.9
        }]

        if user_email not in user_predictions:
            user_predictions[user_email] = []
        user_predictions[user_email].append({
            "item_name": item_name,
            "store_name": store_name,
            "forecast": predictions,
            "suggestions": suggestions,
            "timestamp": datetime.now().isoformat()
        })

        return jsonify({
            "forecast": predictions,
            "suggestions": suggestions,
            "store_name": store_name,
            "item_name": item_name
        }), 200

    except Exception as e:
        logger.error("Error in forecast: %s", str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/predictions', methods=['GET'])
@jwt_required()
def get_predictions():
    user_email = get_jwt_identity()
    logger.info("Fetching predictions for user: %s", user_email)
    predictions = user_predictions.get(user_email, [])
    return jsonify({"predictions": predictions}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)