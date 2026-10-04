import os
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "fraud_detection")

client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
db = client[MONGO_DB]
predictions_collection = db["predictions"]


def is_db_connected() -> bool:
    try:
        client.admin.command("ping")
        return True
    except Exception:
        return False


def save_prediction(
    transaction: dict,
    fraud_probability: float,
    is_fraud: bool,
    threshold_used: float,
    timestamp: datetime,
) -> bool:
    record = {
        "transaction": transaction,
        "fraud_probability": fraud_probability,
        "is_fraud": is_fraud,
        "threshold_used": threshold_used,
        "timestamp": timestamp,
    }
    result = predictions_collection.insert_one(record)
    return result.acknowledged


def get_predictions(limit: int = 20) -> list:
    records = list(
        predictions_collection.find().sort("_id", -1).limit(limit)
    )
    for doc in records:
        doc["_id"] = str(doc["_id"])
    return records
