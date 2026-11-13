
import sqlite3
from pathlib import Path
import bcrypt

DB_PATH = Path(__file__).resolve().parent / "data" / "users.db"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(
        DB_PATH,
        timeout=10
    )

    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def register_user(name, email, password):
    name = name.strip()
    email = email.strip().lower()

    if not name or not email:
        return False, "Name and email are required."

    if len(password.encode("utf-8")) > 72:
        return False, "Password is too long."

    hashed_password = bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt(rounds=12)
    ).decode("utf-8")

    try:
        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO users (name, email, password_hash)
                VALUES (?, ?, ?)
                """,
                (name, email, hashed_password)
            )

        return True, "Account created successfully!"

    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."


def authenticate_user(email, password):
    email = email.strip().lower()

    with get_connection() as connection:
        user = connection.execute(
            """
            SELECT id, name, email, password_hash
            FROM users
            WHERE email = ?
            """,
            (email,)
        ).fetchone()

    if user is None:
        return None

    try:
        password_matches = bcrypt.checkpw(
            password.encode("utf-8"),
            user["password_hash"].encode("utf-8")
        )
    except (ValueError, TypeError):
        return None

    if password_matches:
        return {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"]
        }

    return None
