import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Point this at PostgreSQL in production, e.g.
#   postgresql+psycopg2://postgres:secret@localhost:5432/fraud
# Falls back to a local SQLite file so the project runs with zero setup.
DATABASE_URL = os.getenv("DATABASE_URL") or f"sqlite:///{(ROOT / 'data' / 'fraud.db').as_posix()}"

# LLM provider: ollama (local) | gemini | groq (cloud)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# How many times the generator may re-prompt the model with validation errors.
MAX_GENERATION_ATTEMPTS = int(os.getenv("MAX_GENERATION_ATTEMPTS", "3"))

# Deployment gates applied to every back-test.
BACKTEST_MIN_PRECISION = float(os.getenv("BACKTEST_MIN_PRECISION", "0.30"))
BACKTEST_MAX_FLAG_RATE = float(os.getenv("BACKTEST_MAX_FLAG_RATE", "0.02"))
BACKTEST_MIN_FLAGGED = int(os.getenv("BACKTEST_MIN_FLAGGED", "5"))
