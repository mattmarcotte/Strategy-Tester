# Strategy-Tester

Have fun losing your money

## Local Web UI (FastAPI + HTML)

This project now includes a local browser UI that runs on top of your existing backend modules.

### Start

```bash
pip install -r requirements.txt
uvicorn ui.server:app --reload
```

Open `http://127.0.0.1:8000` in your browser.

### API-first path

- HTML UI route: `GET /`
- Form execution route: `POST /run`
- JSON execution route: `POST /api/run`
- Health route: `GET /api/health`

The `/api/run` endpoint uses the same service function as the HTML UI, so it is ready for a gradual API-first migration later.
