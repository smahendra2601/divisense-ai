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

The pipeline works without RAG, but ingested annual reports give the
forecaster qualitative context (dividend policy, capital allocation).

Place annual report PDFs in `data/annual_reports/`, named
`<TICKER>_<anything>.pdf` (e.g. `ITC_FY2025.pdf`), then ingest them all:

```powershell
python -m src.pdf_ingest --all              # every PDF in the folder
python -m src.pdf_ingest ITC data\annual_reports\ITC_FY2025.pdf   # one file
```

Where to get the PDFs: each company's investor-relations page, or NSE's
public filings API — after visiting nseindia.com once for cookies,
`https://www.nseindia.com/api/annual-reports?index=equities&symbol=<TICKER>`
returns JSON with direct `nsearchives.nseindia.com` PDF links. Re-running
ingestion on the same file updates it in place (no duplicates).

Verify what a ticker's index returns:

```powershell
python -m src.rag ITC "dividend policy"
```

## Tests

```powershell
pytest tests/
```
