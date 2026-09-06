# 🚀 ASTRA-E Web Portal — Team 24-Hour Sprint Guide

**Smart India Hackathon (SIH 26174) — Bhartiya Antariksh Station (BAS)**
**Deadline:** Tomorrow Night

---

## 👥 Team Assignments & File Ownership

| Role | Member | Primary Files |
| :--- | :--- | :--- |
| **Frontend Dev 1** | *[Name 1]* | rontend/src/app/page.tsx, rontend/src/components/hero-section.tsx, rontend/src/components/architecture-diagram.tsx, rontend/src/app/demo/page.tsx, rontend/src/components/demo-player.tsx |
| **Frontend Dev 2** | *[Name 2]* | rontend/src/app/downloads/page.tsx, rontend/src/components/model-card.tsx, rontend/src/app/docs/page.tsx, rontend/src/components/code-block.tsx, rontend/src/components/checksum-verifier.tsx |
| **Backend Dev 1** | *[Name 3]* | ackend/data/models_catalog.json, ackend/api/models.py, ackend/api/downloads.py, ackend/api/checksum.py |
| **Backend Dev 2** | *[Name 4]* | ackend/api/inference.py, ackend/api/telemetry.py, ackend/main.py, rontend/src/app/actions.ts |

---

## 🛠️ How to Run Locally

### 1. Frontend (Next.js 15 + React 19)
`ash
cd astra-portal/frontend
npm install
npm run dev
`
Open [http://localhost:3000](http://localhost:3000)

### 2. Backend (FastAPI)
`ash
cd astra-portal/backend
python -m venv .venv
# On Windows: .venv\Scripts\activate
# On Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
`
API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## ⚡ Zero-Blocker Strategy
Frontend developers should **NOT wait for the backend**. All mock data is ready in rontend/src/lib/mock-data.ts.
Build and polish the UI first! Backend endpoints follow the exact same data structure.
