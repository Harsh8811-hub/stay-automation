# Stay automation

StayInn hotel guest chat automation bot (WhatsApp / booking lifecycle).

## Setup

1. Copy `.env.example` to `.env` and fill secrets.
2. Install deps: `pip install -r requirements.txt`
3. Run: `uvicorn app.main:app --reload` or `.\start.ps1`
4. Or Docker: `docker compose up --build`

## Notes

- Configure MongoDB, LLM keys, and payment backend via environment variables.
- Product branding in this repo is **StayInn**.
