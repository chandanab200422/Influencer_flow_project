from fastapi import FastAPI, HTTPException, Request, Header,Query,UploadFile, File
from pydantic import BaseModel
import requests
import firebase_admin
from firebase_admin import credentials, auth, firestore
import pandas as pd
from database import get_connection, return_connection
from typing import Optional, List
from datetime import date
from faster_whisper import WhisperModel
import os

# === Initialize FastAPI App ===
app = FastAPI()

# === Firebase Setup ===
FIREBASE_API_KEY = "AIzaSyBeo38AA51KuqV9_wBDISAQlITSlUeV60A"  # Replace with your actual API key

cred = credentials.Certificate("servicekey.json")  # Ensure this file exists and is valid
firebase_admin.initialize_app(cred)
db = firestore.client()  # ✅ Firestore client initialized

# === Firebase Auth URLs ===
FIREBASE_SIGNUP_URL = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={FIREBASE_API_KEY}"
FIREBASE_LOGIN_URL = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"

# === Models ===
class AuthRequest(BaseModel):
    email: str
    password: str

class CategoryRequest(BaseModel):
    categories: list[str] = []

# === Root Test Endpoint ===
@app.get("/")
def root():
    return {"message": "Hello World"}

# === Sign Up Endpoint ===
@app.post("/signup")
def signup(auth_data: AuthRequest):
    payload = {
        "email": auth_data.email,
        "password": auth_data.password,
        "returnSecureToken": True
    }
    res = requests.post(FIREBASE_SIGNUP_URL, json=payload)
    if res.status_code != 200:
        raise HTTPException(status_code=400, detail=res.json().get("error", {}).get("message", "Signup failed"))
    return res.json()  # Returns idToken, localId, etc.

# === Login Endpoint ===
@app.post("/login")
def login(auth_data: AuthRequest):
    payload = {
        "email": auth_data.email,
        "password": auth_data.password,
        "returnSecureToken": True
    }
    res = requests.post(FIREBASE_LOGIN_URL, json=payload)
    if res.status_code != 200:
        raise HTTPException(status_code=400, detail=res.json().get("error", {}).get("message", "Login failed"))
    return res.json()  # Returns idToken, localId, etc.

# === Register User Categories (Protected Endpoint) ===
@app.post("/register-user")
def register_user(request: CategoryRequest, authorization: str = Header(...)):
    try:
        # Extract token from Authorization header
        id_token = authorization.split(" ")[1]
        decoded_token = auth.verify_id_token(id_token)
        user_id = decoded_token["uid"]

        selected_categories = request.categories if request.categories else ["not given"]

        # Store in Firestore
        db.collection("users").document(user_id).set({
            "categories": selected_categories
        })

        return {"message": "User registered with categories", "categories": selected_categories}

    except Exception as e:
        print("❌ Error in /register-user:", e)
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/creators")
def get_creators(
    location: Optional[List[str]] = Query(None),   # multi-select
    niche: Optional[List[str]] = Query(None),      # multi-select
    platforms: Optional[List[str]] = Query(None),  # multi-select

    followers_min: Optional[int] = None,
    followers_max: Optional[int] = None,

    avg_views_min: Optional[int] = None,
    avg_views_max: Optional[int] = None,

    product_sales_min: Optional[float] = None,  # if numeric, else skip
    product_sales_max: Optional[float] = None,  # else treat as text or skip

    product_price_min: Optional[float] = None,
    product_price_max: Optional[float] = None,

    engagement_rate_min: Optional[float] = None,
    engagement_rate_max: Optional[float] = None,

    product_rating_min: Optional[float] = None,
    product_rating_max: Optional[float] = None,

    endorsement_date_start: Optional[date] = None,
    endorsement_date_end: Optional[date] = None,

    hashtags: Optional[str] = None,   # simple text match, e.g. '#fashion'
    languages: Optional[str] = None   # text match, e.g. 'English'
):
    conn = get_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM creators WHERE TRUE"
    params = []

    # Multi-select filters (IN with ILIKE)
    if location:
        query += " AND (" + " OR ".join(["location ILIKE %s"] * len(location)) + ")"
        params.extend([f"%{loc}%" for loc in location])

    if niche:
        query += " AND (" + " OR ".join(["niche ILIKE %s"] * len(niche)) + ")"
        params.extend([f"%{n}%" for n in niche])

    if platforms:
        query += " AND (" + " OR ".join(["platforms ILIKE %s"] * len(platforms)) + ")"
        params.extend([f"%{p}%" for p in platforms])

    # Range filters
    if followers_min is not None:
        query += " AND followers >= %s"
        params.append(followers_min)
    if followers_max is not None:
        query += " AND followers <= %s"
        params.append(followers_max)

    if avg_views_min is not None:
        query += " AND avg_views >= %s"
        params.append(avg_views_min)
    if avg_views_max is not None:
        query += " AND avg_views <= %s"
        params.append(avg_views_max)

    # If product_sales is numeric (you must confirm this!)
    if product_sales_min is not None:
        query += " AND product_sales::numeric >= %s"
        params.append(product_sales_min)
    if product_sales_max is not None:
        query += " AND product_sales::numeric <= %s"
        params.append(product_sales_max)

    if product_price_min is not None:
        query += " AND product_price >= %s"
        params.append(product_price_min)
    if product_price_max is not None:
        query += " AND product_price <= %s"
        params.append(product_price_max)

    # Threshold / slider filters
    if engagement_rate_min is not None:
        query += " AND engagement_rate >= %s"
        params.append(engagement_rate_min)
    if engagement_rate_max is not None:
        query += " AND engagement_rate <= %s"
        params.append(engagement_rate_max)

    if product_rating_min is not None:
        query += " AND product_rating >= %s"
        params.append(product_rating_min)
    if product_rating_max is not None:
        query += " AND product_rating <= %s"
        params.append(product_rating_max)

    # Date range
    if endorsement_date_start is not None:
        query += " AND endorsement_date >= %s"
        params.append(endorsement_date_start)
    if endorsement_date_end is not None:
        query += " AND endorsement_date <= %s"
        params.append(endorsement_date_end)

    # Text/tag matching (simple ILIKE)
    if hashtags:
        query += " AND hashtags ILIKE %s"
        params.append(f"%{hashtags}%")
    if languages:
        query += " AND languages ILIKE %s"
        params.append(f"%{languages}%")

    cursor.execute(query, tuple(params))
    rows = cursor.fetchall()

    columns = [desc[0] for desc in cursor.description]
    data = [dict(zip(columns, row)) for row in rows]

    cursor.close()
    return_connection(conn)

    return {"creators": data}

def rule_based_analysis(creator: dict[str, any], product: dict[str, any]) -> dict[str, any]:
    score = 0.0

    # Niche match
    if product["category"].lower() in creator["niche"].lower():
        score += 0.2

    # High engagement
    if creator["engagement_rate"] >= 5:
        score += 0.2

    # High average views
    if creator["avg_views"] >= 500000:
        score += 0.2

    # Price match (±₹500)
    try:
        if abs(product["price"] - creator["product_price"]) <= 500:
            score += 0.1
    except:
        pass

    # Language match
    if any(lang in creator["languages"] for lang in ["English", "Hindi"]):
        score += 0.1

    # Platform match
    if any(platform in creator["platforms"] for platform in ["Instagram", "YouTube"]):
        score += 0.1

    # Past product success
    try:
        sales = creator["product_sales"].lower().replace("l", "")
        if float(sales) >= 10:
            score += 0.1
    except:
        pass

    estimated_reach = int(creator["avg_views"] * (score + 0.1))  # 0.1 base boost

    return {
        "creator_id": creator["creator_id"],
        "username": creator["username"],
        "score": round(score, 2),
        "estimated_reach": estimated_reach,
        "platforms": creator["platforms"],
        "niche": creator["niche"],
        "languages": creator["languages"],
        "profile_pic": creator["profile_pic"],
        "product_sales": creator["product_sales"],
        "engagement_rate": creator["engagement_rate"],
        "reason": "Rule-based scoring on views, engagement, match, and past performance"
    }


class Creator(BaseModel):
    creator_id: int
    name: str
    username: str
    bio: str
    location: str
    niche: str
    followers: int
    engagement_rate: float
    avg_views: int
    avg_likes: int
    avg_comments: int
    best_product: str
    product_sales: str
    product_category: str
    product_rating: float
    product_price: float
    product_image: str
    endorsement_type: str
    endorsement_date: str
    languages: str
    profile_pic: str
    platforms: str
    hashtags: str

class Product(BaseModel):
    name: str
    category: str
    price: float
    rating: float

@app.post("/analyse-promotion")
def analyse_promotion(creators: List[Creator], product: Product):
    results = []
    for creator in creators:
        result = rule_based_analysis(creator.dict(), product.dict())
        results.append(result)
    return {"recommendations": sorted(results, key=lambda x: x["score"], reverse=True)}

# Updated chatbot logic with better error handling
def get_whisper_model():
    return WhisperModel("small", device="cpu", compute_type="int8")

@app.post("/chatbot/voice")
async def chatbot_voice(file: UploadFile = File(...)):
    try:
        # Save audio temporarily
        with open("audio1.mp3", "wb") as f:
            f.write(await file.read())
        
        # Transcribe
        model = get_whisper_model()
        segments, _ = model.transcribe("audio1.mp3")
        transcript = " ".join(segment.text for segment in segments)
        
        return {"transcription": transcript}
    finally:
        if os.path.exists("audio1.mp3"):
            os.remove("audio1.mp3")

import uuid,re

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import uuid
import re
import random
from enum import Enum
from datetime import datetime

app = FastAPI()

# Session storage
sessions = {}

# Models
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    filters: Optional[Dict] = None

class Creator(BaseModel):
    creator_id: int
    name: str
    username: str
    niche: str
    followers: int
    engagement_rate: float
    avg_views: int
    languages: str
    platforms: str
    profile_pic: str
    product_sales: str
    product_rating: float
    product_price: float
    email: Optional[str] = None
    phone: Optional[str] = None

class Product(BaseModel):
    name: str
    category: str
    price: float
    rating: float

class Category(str, Enum):
    FASHION = "fashion"
    BEAUTY = "beauty"
    FITNESS = "fitness"
    TECH = "technology"
    FOOD = "food"
    TRAVEL = "travel"

# Helper functions
def analyze_category(product_desc: str) -> Category:
    """Categorize product using AI (mock)"""
    product_desc = product_desc.lower()
    if any(x in product_desc for x in ["makeup", "skincare"]):
        return Category.BEAUTY
    elif any(x in product_desc for x in ["dress", "clothing"]):
        return Category.FASHION
    elif any(x in product_desc for x in ["camera", "tech"]):
        return Category.TECH
    return Category.FASHION

def get_filtered_creators(category: str, filters: Dict) -> List[Creator]:
    """Mock DB query - replace with actual /creators call"""
    # This would be your actual API call:
    # response = requests.get("http://localhost:8000/creators", params=filters)
    mock_creators = [
        Creator(
            creator_id=1,
            name="Ashtrita",
            username="ashtrita_beauty",
            niche="beauty",
            followers=150000,
            engagement_rate=4.5,
            avg_views=200000,
            languages="English, Hindi",
            platforms="Instagram, YouTube",
            profile_pic="url1",
            product_sales="50K",
            product_rating=4.7,
            product_price=6000,
            email="ashtrita@example.com"
        ),
        Creator(
            creator_id=2,
            name="Riya",
            username="riyaglow",
            niche="beauty",
            followers=80000,
            engagement_rate=4.2,
            avg_views=150000,
            languages="English",
            platforms="Instagram",
            profile_pic="url2",
            product_sales="30K",
            product_rating=4.5,
            product_price=4500
        )
    ]
    return [c for c in mock_creators if c.product_price <= filters.get("max_price", float('inf'))]

def analyze_promotion(creators: List[Creator], product: Product) -> List[Dict]:
    """Your existing scoring function"""
    results = []
    for creator in creators:
        score = 0.0
        if product.category.lower() in creator.niche.lower():
            score += 0.3
        if creator.engagement_rate > 4:
            score += 0.2
        if abs(product.price - creator.product_price) <= (0.2 * product.price):
            score += 0.1
        if "english" in creator.languages.lower():
            score += 0.1
            
        results.append({
            **creator.dict(),
            "score": round(score, 2),
            "estimated_reach": int(creator.avg_views * (score + 0.1))
        })
    return sorted(results, key=lambda x: x["score"], reverse=True)

def generate_content(product: Dict, creator: Dict) -> str:
    """AI-generated content (mock)"""
    return f"""🌟 {creator['name']} recommends {product['name']}!
    
Perfect for #{product['category']} lovers!
👉 Special collaboration offer: 10% off with code {creator['username'].upper()}10"""

def contact_creator(creator: Creator, content: str) -> bool:
    """Mock contact function"""
    print(f"Emailing {creator.email}:\n{content}")
    return True

def negotiate(creator: Creator, user_offer: float) -> Optional[float]:
    """Smart negotiation logic"""
    creator_min = creator.product_price * 0.85  # Creators won't go below 15% discount
    
    if user_offer >= creator_min:
        # 60% chance to accept if within 15% of their rate
        return None if random.random() < 0.6 else creator_min
    return max(user_offer * 1.1, creator_min)  # Counter with 10% increase or their minimum

@app.post("/chat")
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    message = req.message.lower().strip()
    filters = req.filters or {}

    # Initialize session
    if session_id not in sessions:
        sessions[session_id] = {
            "state": "ask_product",
            "product": None,
            "category": None,
            "budget": None,
            "creators": None,
            "selected_creator": None,
            "negotiation_attempts": 0,
            "content_approved": False,
            "contact_attempts": 0,
            "content": None,
            "filters": {"max_price": filters.get("max_price")}
        }

    state = sessions[session_id]["state"]
    session = sessions[session_id]

    # State handlers
    if state == "ask_product":
        session["product"] = message
        session["category"] = analyze_category(message)
        session["state"] = "ask_budget"
        return {
            "session_id": session_id,
            "reply": f"🔍 Identified as {session['category'].value} category. What's your budget?",
            "done": False
        }

    elif state == "ask_budget":
        if match := re.search(r"(\d+)", message):
            session["budget"] = float(match.group(1))
            session["filters"]["max_price"] = session["budget"] * 1.2  # Show slightly higher options
            
            try:
                creators = get_filtered_creators(
                    session["category"].value,
                    session["filters"]
                )
                
                if not creators:
                    return {
                        "session_id": session_id,
                        "reply": "🚫 No creators match your criteria. Try increasing budget or say 'adjust filters'",
                        "done": False
                    }
                
                product = Product(
                    name=session["product"],
                    category=session["category"].value,
                    price=session["budget"],
                    rating=4.0
                )
                
                session["creators"] = analyze_promotion(creators, product)
                session["state"] = "select_creator"
                
                creators_list = "\n".join(
                    f"{i+1}. {c['name']} (₹{c['product_price']:,}) | "
                    f"Score: {c['score']:.2f} | Reach: {c['estimated_reach']:,}"
                    for i, c in enumerate(session["creators"][:5]))
                
                return {
                    "session_id": session_id,
                    "reply": f"🏆 Top Creators:\n{creators_list}\n\n"
                             "Select a number or say 'adjust filters'",
                    "done": False
                }
            except Exception as e:
                return error_response(session_id, str(e))

        return {
            "session_id": session_id,
            "reply": "💳 Please enter a valid budget amount (e.g. 5000)",
            "done": False
        }

    elif state == "select_creator":
        if "adjust" in message:
            session["state"] = "adjust_filters"
            return {
                "session_id": session_id,
                "reply": "🔧 Current filters:\n"
                         f"- Max price: ₹{session['filters']['max_price']:,}\n\n"
                         "Send new max price or say 'skip'",
                "done": False
            }
        
        try:
            selection = int(message) - 1
            if 0 <= selection < len(session["creators"]):
                session["selected_creator"] = selected = session["creators"][selection]
                
                if selected["product_price"] > session["budget"]:
                    session["state"] = "handle_high_quote"
                    return {
                        "session_id": session_id,
                        "reply": f"⚠️ {selected['name']}'s rate is ₹{selected['product_price']:,} "
                                 f"(Your budget: ₹{session['budget']:,})\n\n"
                                 "Should I:\n1. Negotiate\n2. Suggest alternatives\n3. Increase budget",
                        "done": False
                    }
                
                session["content"] = generate_content(
                    {"name": session["product"], "category": session["category"].value},
                    selected
                )
                session["state"] = "approve_content"
                return {
                    "session_id": session_id,
                    "reply": f"📝 Content draft for {selected['name']}:\n\n{session['content']}\n\n"
                             "Approve? (yes/no/edit)",
                    "done": False
                }
        except ValueError:
            pass
        
        return {
            "session_id": session_id,
            "reply": "🔢 Please select a valid creator number",
            "done": False
        }

    elif state == "handle_high_quote":
        if "1" in message or "negotiate" in message:
            session["negotiation_attempts"] = 1
            initial_offer = min(
                session["budget"],
                session["selected_creator"]["product_price"] * 0.9
            )
            session["current_offer"] = initial_offer
            session["state"] = "negotiating"
            return {
                "session_id": session_id,
                "reply": f"✍️ Negotiating with {session['selected_creator']['name']}...\n"
                         f"Initial offer: ₹{initial_offer:,}",
                "done": False
            }
        
        elif "2" in message or "alternative" in message:
            session["state"] = "select_creator"
            return {
                "session_id": session_id,
                "reply": "Showing alternative creators...",
                "done": False
            }
        
        elif "3" in message or "increase" in message:
            session["state"] = "increase_budget"
            return {
                "session_id": session_id,
                "reply": "💵 Enter new budget amount:",
                "done": False
            }

    elif state == "negotiating":
        creator = session["selected_creator"]
        current_offer = session["current_offer"]
        
        if session["negotiation_attempts"] >= 3:
            session["state"] = "negotiation_failed"
            return {
                "session_id": session_id,
                "reply": f"❌ Couldn't reach agreement with {creator['name']}\n"
                         "Options:\n1. Accept final offer\n2. Try others\n3. Cancel",
                "done": False
            }
        
        # Simulate creator response
        counter_offer = negotiate(creator, current_offer)
        
        if counter_offer is None:  # Accepted
            session["state"] = "deal_confirmed"
            return {
                "session_id": session_id,
                "reply": f"✅ {creator['name']} accepted ₹{current_offer:,}!",
                "done": True
            }
        
        session["current_offer"] = counter_offer
        session["negotiation_attempts"] += 1
        
        return {
            "session_id": session_id,
            "reply": f"📈 {creator['name']} counters with ₹{counter_offer:,}\n"
                     f"Attempt {session['negotiation_attempts']}/3\n\n"
                     "Reply with:\n- New offer amount\n- 'accept' to agree\n- 'walk away'",
            "done": False
        }

    # [Additional state handlers for content approval, etc...]

    return {
        "session_id": session_id,
        "reply": "I didn't understand that. Please try again.",
        "done": False
    }

def error_response(session_id: str, error: str) -> Dict:
    return {
        "session_id": session_id,
        "reply": f"⚠️ Error: {error}",
        "done": True
    }