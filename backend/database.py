import sqlite3
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

DB_PATH = Path(__file__).resolve().parent / "chat_storage.db"


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL DEFAULT 'demo_user',
                title TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                medical_references TEXT DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
        """)

        # Prediction Records table for grounding AI with actual user ML predictions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL DEFAULT 'demo_user',
                risk_tier TEXT NOT NULL,
                cluster INTEGER,
                risk_model TEXT,
                predicted_premium REAL NOT NULL,
                global_model TEXT,
                global_prediction REAL NOT NULL,
                diff REAL,
                top_factors TEXT,
                customer_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


# Initialize on import
init_db()


def create_conversation(user_id: str = "demo_user", title: str = "New Consultation") -> Dict[str, Any]:
    conv_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO conversations (id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (conv_id, user_id, title, now, now)
        )
        conn.commit()
    return {"id": conv_id, "user_id": user_id, "title": title, "created_at": now, "updated_at": now}


def get_user_conversations(user_id: str = "demo_user") -> List[Dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT c.id, c.user_id, c.title, c.created_at, c.updated_at,
                   (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) as message_count,
                   (SELECT m.content FROM messages m WHERE m.conversation_id = c.id ORDER BY m.created_at DESC LIMIT 1) as last_message
            FROM conversations c
            WHERE c.user_id = ?
            ORDER BY c.updated_at DESC
            """,
            (user_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_conversation(conv_id: str, user_id: str = "demo_user") -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM conversations WHERE id = ? AND user_id = ?",
            (conv_id, user_id)
        ).fetchone()
        return dict(row) if row else None


def get_conversation_any_user(conv_id: str) -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM conversations WHERE id = ?",
            (conv_id,)
        ).fetchone()
        return dict(row) if row else None



def update_conversation_title(conv_id: str, title: str):
    with get_db() as conn:
        conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, datetime.now(timezone.utc).isoformat(), conv_id)
        )
        conn.commit()


def delete_conversation(conv_id: str, user_id: str = "demo_user") -> bool:
    with get_db() as conn:
        # Check ownership
        row = conn.execute("SELECT id FROM conversations WHERE id = ? AND user_id = ?", (conv_id, user_id)).fetchone()
        if not row:
            return False
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
        conn.commit()
        return True


def add_message(
    conversation_id: str,
    role: str,
    content: str,
    medical_references: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    msg_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    refs_json = json.dumps(medical_references or [])
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO messages (id, conversation_id, role, content, medical_references, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (msg_id, conversation_id, role, content, refs_json, now)
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id)
        )
        conn.commit()
    return {
        "id": msg_id,
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
        "medical_references": medical_references or [],
        "created_at": now
    }


def get_messages(conversation_id: str) -> List[Dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
            (conversation_id,)
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            try:
                d["medical_references"] = json.loads(d.get("medical_references") or "[]")
            except Exception:
                d["medical_references"] = []
            result.append(d)
        return result


def save_prediction(
    user_id: str,
    prediction_result: Dict[str, Any],
    customer_data: Optional[Dict[str, Any]] = None
) -> str:
    pred_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO predictions (
                id, user_id, risk_tier, cluster, risk_model,
                predicted_premium, global_model, global_prediction, diff,
                top_factors, customer_data, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pred_id,
                user_id,
                prediction_result.get("risk", "Medium"),
                prediction_result.get("cluster", 0),
                prediction_result.get("risk_model", "Unknown"),
                prediction_result.get("risk_aware_prediction", 0.0),
                prediction_result.get("global_model", "RandomForest"),
                prediction_result.get("global_prediction", 0.0),
                prediction_result.get("diff", 0.0),
                json.dumps(prediction_result.get("top_shap_factors", [])),
                json.dumps(customer_data or {}),
                now
            )
        )
        conn.commit()
    return pred_id


def get_latest_prediction(user_id: str = "demo_user") -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
            (user_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        try:
            d["top_factors"] = json.loads(d.get("top_factors") or "[]")
            d["customer_data"] = json.loads(d.get("customer_data") or "{}")
        except Exception:
            d["top_factors"] = []
            d["customer_data"] = {}
        return d


def get_prediction_history(user_id: str = "demo_user", limit: int = 10) -> List[Dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit)
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["top_factors"] = json.loads(d.get("top_factors") or "[]")
                d["customer_data"] = json.loads(d.get("customer_data") or "{}")
            except Exception:
                d["top_factors"] = []
                d["customer_data"] = {}
            out.append(d)
        return out
