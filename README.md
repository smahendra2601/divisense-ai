# DiviSense AI

Agentic dividend forecasting for NSE-listed Indian companies. Ask a
natural-language question ("Will Infosys increase its dividend next
quarter?") or enter a ticker ("ITC") and get a next-FY dividend
forecast with transparent reasoning, confidence, and risks.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design (source of truth).

> ⚠️ Research tool, not investment advice.

## Setup (Windows / PowerShell)

```powershell
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate it
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
copy .env.example .env
# then edit .env and fill in GROQ_API_KEY and GOOGLE_API_KEY
```

macOS / Linux equivalent:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in the keys
```

## Run

```powershell
# Streamlit UI
streamlit run app.py

# CLI
python forecast.py "Will Infosys increase its dividend next quarter?"
python forecast.py ITC
```

## One-time RAG ingestion (optional)

Place annual report PDFs in `data/annual_reports/`, then:

```powershell
python src/pdf_ingest.py
```

## Tests

```powershell
pytest tests/
```
