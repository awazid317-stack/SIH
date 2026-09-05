# ============================================================
# AI LEGAL ASSISTANT API — COMPLETE SINGLE-CELL GOOGLE COLAB CODE
# ============================================================
# Steps:
# 1) Change NGROK_AUTHTOKEN below with your NEW ngrok token.
# 2) Run this cell.
# 3) Upload legal_advice_database.csv when prompted.
# 4) Open the printed Swagger Docs URL: https://....ngrok-free.app/docs
# ============================================================

!pip install -q fastapi "uvicorn[standard]" pyngrok pandas requests nest_asyncio

import os
import time
import threading
from datetime import datetime

import pandas as pd
import requests
import nest_asyncio
import uvicorn

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from google.colab import files
from pyngrok import ngrok

# ------------------------------------------------------------
# 1. PUT YOUR NEW NGROK AUTH TOKEN HERE
# ------------------------------------------------------------
NGROK_AUTHTOKEN = "3Iuyknk8zjnTWloF7JUvArj4KHL_6WM8HGdEG3ezRqbc8nUEq"

if not NGROK_AUTHTOKEN or NGROK_AUTHTOKEN == "PASTE_NEW_TOKEN_HERE":
    raise ValueError(
        "❌ NGROK_AUTHTOKEN paste karo. "
        "Ngrok dashboard se NEW token generate karke yahan daalo."
    )

# Colab event-loop compatibility
nest_asyncio.apply()

# ------------------------------------------------------------
# 2. UPLOAD CSV FILE
# ------------------------------------------------------------
print("📁 Select legal_advice_database.csv to upload:")
uploaded = files.upload()

CSV_FILE = "legal_advice_database.csv"

# Agar uploaded filename different ho, first uploaded file use kar lo
if CSV_FILE not in uploaded:
    if len(uploaded) == 0:
        raise FileNotFoundError("❌ Koi CSV upload nahi hui.")
    CSV_FILE = list(uploaded.keys())[0]
    print(f"⚠️ Using uploaded file: {CSV_FILE}")

# ------------------------------------------------------------
# 3. LOAD AND VALIDATE DATABASE
# ------------------------------------------------------------
df = pd.read_csv(CSV_FILE)

required_columns = [
    "query",
    "category",
    "ipc_sections",
    "advice",
    "recommended_action",
    "severity",
    "estimated_time"
]

missing_columns = [column for column in required_columns if column not in df.columns]

if missing_columns:
    raise ValueError(
        f"❌ CSV mein required columns missing hain: {missing_columns}\n\n"
        f"Available columns: {list(df.columns)}\n\n"
        "CSV mein ye columns hone chahiye:\n"
        "query, category, ipc_sections, advice, "
        "recommended_action, severity, estimated_time"
    )

df = df.dropna(subset=["category"]).copy()
df["category"] = df["category"].astype(str).str.strip()

print(f"\n✅ Loaded {len(df)} legal cases")
print(f"✅ Categories: {list(df['category'].unique())}")

# ------------------------------------------------------------
# 4. LEGAL CATEGORY CLASSIFICATION
# ------------------------------------------------------------
def classify_query(query: str):
    category_keywords = {
        "Criminal": [
            "crime", "police", "fir", "violence", "maarta",
            "threatening", "assault", "maar-peet", "murder",
            "theft", "attack", "kidnap", "criminal"
        ],
        "Labor": [
            "salary", "job", "boss", "employment", "pension",
            "company", "workplace", "wage", "employee",
            "office", "resign", "termination"
        ],
        "Consumer": [
            "product", "defective", "quality", "refund", "bank",
            "insurance", "online shopping", "delivery", "seller",
            "consumer", "replacement", "fake product"
        ],
        "Family": [
            "divorce", "marriage", "custody", "maintenance",
            "domestic", "wife", "husband", "alimony", "child",
            "dowry", "family"
        ],
        "Civil": [
            "dispute", "contract", "boundary", "damages", "suit",
            "agreement", "neighbour", "compensation", "civil"
        ],
        "Property": [
            "land", "registration", "rent", "flat", "ghar",
            "building", "tenant", "house", "plot", "owner",
            "property", "lease"
        ]
    }

    query_lower = query.lower().strip()

    match_counts = {
        category: sum(keyword in query_lower for keyword in keywords)
        for category, keywords in category_keywords.items()
    }

    best_category = max(match_counts, key=match_counts.get)
    matches = match_counts[best_category]
    total_keywords = len(category_keywords[best_category])

    confidence = round((matches / total_keywords) * 100, 2) if matches > 0 else 0.0

    return {
        "category": best_category,
        "confidence": confidence,
        "matched_keywords": matches
    }

# ------------------------------------------------------------
# 5. RETRIEVE RELATED LEGAL ADVICE FROM CSV
# ------------------------------------------------------------
def get_legal_advice(category: str, top_k: int = 2):
    category_cases = df[
        df["category"].astype(str).str.lower() == category.lower()
    ]

    if category_cases.empty:
        return {
            "ipc": [],
            "advice": "Please consult a qualified lawyer for case-specific legal guidance.",
            "action": "Seek legal counsel",
            "severity": "Unknown",
            "time": "Unknown"
        }

    relevant_cases = category_cases.head(top_k)
    first_case = relevant_cases.iloc[0]

    ipc_sections = (
        relevant_cases["ipc_sections"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()[:3]
    )

    advice_values = (
        relevant_cases["advice"]
        .dropna()
        .astype(str)
        .tolist()
    )

    return {
        "ipc": ipc_sections,
        "advice": advice_values[0] if advice_values else "Consult a qualified lawyer.",
        "action": str(first_case.get("recommended_action", "Consult lawyer")),
        "severity": str(first_case.get("severity", "Unknown")),
        "time": str(first_case.get("estimated_time", "Unknown"))
    }

# ------------------------------------------------------------
# 6. COMPLETE LEGAL QUERY PIPELINE
# ------------------------------------------------------------
def process_legal_query(query: str):
    query = query.strip()

    if not query:
        raise ValueError("Query empty nahi ho sakti.")

    classification = classify_query(query)
    legal_info = get_legal_advice(classification["category"])

    return {
        "query": query,
        "category": classification["category"],
        "confidence": classification["confidence"],
        "ipc_sections": legal_info["ipc"],
        "legal_advice": legal_info["advice"],
        "recommended_action": legal_info["action"],
        "severity": legal_info["severity"],
        "estimated_time": legal_info["time"]
    }

# ------------------------------------------------------------
# 7. CREATE FASTAPI APP
# ------------------------------------------------------------
app = FastAPI(
    title="AI Legal Assistant — SIH 2026",
    version="1.0.0",
    description="Legal query category classification and database retrieval API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def root():
    return {
        "status": "running",
        "message": "AI Legal Assistant API is running",
        "health_endpoint": "/health",
        "categories_endpoint": "/categories",
        "documentation": "/docs"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "legal_cases_loaded": int(len(df))
    }

@app.get("/categories")
def get_categories():
    categories = sorted(df["category"].dropna().unique().tolist())

    return {
        "categories": categories,
        "total_categories": len(categories)
    }

@app.post("/predict")
def predict_legal_category(
    query: str = Query(
        ...,
        min_length=2,
        description="Enter a legal query in English, Hindi, or Hinglish"
    )
):
    try:
        result = process_legal_query(query)

        return {
            "status": "success",
            "query": result["query"],
            "category": result["category"],
            "confidence": result["confidence"],
            "ipc_sections": result["ipc_sections"],
            "legal_advice": result["legal_advice"],
            "recommended_action": result["recommended_action"],
            "severity": result["severity"],
            "estimated_resolution_time": result["estimated_time"],
            "timestamp": datetime.now().isoformat(),
            "disclaimer": (
                "This API provides general informational output only. "
                "It is not a substitute for advice from a qualified lawyer."
            )
        }

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))

print("\n✅ FastAPI app created")

# ------------------------------------------------------------
# 8. START FASTAPI SERVER
# ------------------------------------------------------------
def run_server():
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="warning"
    )

print("🚀 Starting local FastAPI server...")

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

time.sleep(3)

# First test local server
try:
    local_response = requests.get(
        "http://127.0.0.1:8000/health",
        timeout=20
    )

    if local_response.status_code != 200:
        raise RuntimeError(local_response.text)

    print("✅ Local server working")
    print("Local health:", local_response.json())

except Exception as error:
    raise RuntimeError(f"❌ Local FastAPI server start nahi hua: {error}")

# ------------------------------------------------------------
# 9. CREATE NGROK PUBLIC URL
# ------------------------------------------------------------
print("\n🌐 Creating ngrok public URL...")

try:
    ngrok.kill()
except Exception:
    pass

ngrok.set_auth_token(NGROK_AUTHTOKEN)

tunnel = ngrok.connect(
    addr=8000,
    proto="http"
)

API_URL = tunnel.public_url.rstrip("/")

print("\n" + "=" * 70)
print("✅✅✅ API IS LIVE ON THE WEB ✅✅✅")
print("=" * 70)
print(f"\n🌐 Main URL:       {API_URL}")
print(f"❤️  Health URL:     {API_URL}/health")
print(f"📚 Swagger Docs:   {API_URL}/docs")
print(f"📂 Categories:     {API_URL}/categories")
print(f"🤖 Predict API:    {API_URL}/predict")
print("\n👉 OPEN THIS IN BROWSER:")
print(f"{API_URL}/docs")
print("=" * 70)

# ------------------------------------------------------------
# 10. AUTOMATIC PUBLIC API TEST
# ------------------------------------------------------------
print("\n🧪 Testing public API...\n")

ngrok_headers = {
    "ngrok-skip-browser-warning": "true"
}

try:
    public_health = requests.get(
        f"{API_URL}/health",
        headers=ngrok_headers,
        timeout=30
    )

    print("Public health status:", public_health.status_code)
    print("Public health response:", public_health.json())

    if public_health.status_code != 200:
        raise RuntimeError(public_health.text)

    test_query = "Mere boss ne 3 mahine se salary nahi di"

    test_response = requests.post(
        f"{API_URL}/predict",
        params={"query": test_query},
        headers=ngrok_headers,
        timeout=30
    )

    print("\nTest query:", test_query)
    print("Predict status:", test_response.status_code)

    if test_response.status_code == 200:
        print("✅ Public prediction successful!")
        print(test_response.json())
    else:
        print("❌ Predict error:", test_response.text[:1000])

except Exception as error:
    print(f"❌ Public API test failed: {error}")

print("\n🎉 DONE!")
print(f"Browser mein ye open karo: {API_URL}/docs")
print("Swagger page mein POST /predict → Try it out → query enter → Execute.")