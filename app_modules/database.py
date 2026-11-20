"""
Database Module
===============
Manages local SQLite storage for prediction history in prediction_history.db.
Provides functions to insert, query, filter, and clear historical grading records.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import sqlite3
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "prediction_history.db"


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Creates a connection to SQLite database."""
    target_path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(str(target_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """Initializes the prediction history database schema."""
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prediction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    fruit TEXT NOT NULL,
                    quality TEXT NOT NULL,
                    class_label TEXT NOT NULL,
                    probability_score REAL NOT NULL,
                    mean_hue REAL,
                    mean_saturation REAL,
                    mean_brightness REAL,
                    contrast REAL,
                    homogeneity REAL,
                    energy REAL,
                    correlation REAL
                )
            """)
    finally:
        conn.close()


def save_prediction(
    filename: str,
    fruit: str,
    quality: str,
    class_label: str,
    probability_score: float,
    feature_dict: Optional[Dict[str, float]] = None,
    db_path: Optional[Path] = None,
    custom_timestamp: Optional[str] = None
) -> int:
    """
    Saves a successful prediction record into the database.

    Returns:
        int: The inserted record ID.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    timestamp = custom_timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    feats = feature_dict or {}

    try:
        with conn:
            cursor = conn.execute("""
                INSERT INTO prediction_history (
                    timestamp, filename, fruit, quality, class_label,
                    probability_score, mean_hue, mean_saturation, mean_brightness,
                    contrast, homogeneity, energy, correlation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp,
                filename,
                fruit,
                quality,
                class_label,
                float(probability_score),
                float(feats.get("mean_hue", 0.0)),
                float(feats.get("mean_saturation", 0.0)),
                float(feats.get("mean_brightness", 0.0)),
                float(feats.get("contrast", 0.0)),
                float(feats.get("homogeneity", 0.0)),
                float(feats.get("energy", 0.0)),
                float(feats.get("correlation", 0.0)),
            ))
            return cursor.lastrowid
    finally:
        conn.close()


def get_all_predictions(
    limit: int = 100,
    fruit_filter: Optional[str] = None,
    quality_filter: Optional[str] = None,
    db_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Retrieves prediction history records as a pandas DataFrame.
    """
    init_db(db_path)
    conn = get_connection(db_path)

    query = "SELECT * FROM prediction_history WHERE 1=1"
    params = []

    if fruit_filter and fruit_filter != "All":
        query += " AND fruit = ?"
        params.append(fruit_filter)

    if quality_filter and quality_filter != "All":
        query += " AND quality = ?"
        params.append(quality_filter)

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    try:
        df = pd.read_sql_query(query, conn, params=params)
        return df
    except Exception as e:
        print(f"Database query error: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def clear_history(db_path: Optional[Path] = None) -> bool:
    """Deletes all records from the prediction history table."""
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute("DELETE FROM prediction_history")
            conn.execute("DELETE FROM sqlite_sequence WHERE name='prediction_history'")
        return True
    except Exception as e:
        print(f"Error clearing history: {e}")
        return False
    finally:
        conn.close()


def get_statistics(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Computes summary statistics on prediction history."""
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        with conn:
            total = conn.execute("SELECT COUNT(*) FROM prediction_history").fetchone()[0]
            fresh = conn.execute("SELECT COUNT(*) FROM prediction_history WHERE quality='Fresh'").fetchone()[0]
            rotten = conn.execute("SELECT COUNT(*) FROM prediction_history WHERE quality='Rotten'").fetchone()[0]

            top_fruit_row = conn.execute(
                "SELECT fruit, COUNT(*) as c FROM prediction_history GROUP BY fruit ORDER BY c DESC LIMIT 1"
            ).fetchone()
            top_fruit = top_fruit_row[0] if top_fruit_row else "None"

        return {
            "total_predictions": total,
            "fresh_count": fresh,
            "rotten_count": rotten,
            "fresh_ratio": round((fresh / total * 100), 1) if total > 0 else 0.0,
            "most_tested_fruit": top_fruit,
        }
    except Exception as e:
        print(f"Error computing statistics: {e}")
        return {
            "total_predictions": 0,
            "fresh_count": 0,
            "rotten_count": 0,
            "fresh_ratio": 0.0,
            "most_tested_fruit": "None",
        }
    finally:
        conn.close()
