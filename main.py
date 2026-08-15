"""
main.py
Iska kaam: WhatsApp se aane wale messages ko receive karna,
AI (chat.py) se jawab lena, aur wapas WhatsApp pe reply bhejna.

Ye ek FastAPI "web server" hai - matlab ye 24/7 chalke internet
se requests sunta rehta hai (jab tak hum ise chala ke rakhein).
"""

import os
import requests
from fastapi import FastAPI, Request
from dotenv import load_dotenv

# Humara pehle se bana AI function yaha se import kar rahe hain
from chat import get_ai_response

load_dotenv()

META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
META_PHONE_NUMBER_ID = os.getenv("META_PHONE_NUMBER_ID")

# Webhook verify karne ke liye ek secret word - khud choose karo,
# isko Meta portal me bhi wahi likhna hoga jo yaha likhoge
VERIFY_TOKEN = "mera_secret_token_123"

app = FastAPI()


@app.get("/webhook")
async def verify_webhook(request: Request):
    """
    Meta jab pehli baar webhook connect karta hai, to ye endpoint
    call karke check karta hai ki ye tumhara hi server hai ya nahi.
    Isko sirf EK BAAR call hota hai, jab tum Meta portal me
    webhook URL save karte ho.
    """
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        print("✅ Webhook verify ho gaya!")
        return int(challenge)
    return {"error": "Verification failed"}


@app.post("/webhook")
async def receive_message(request: Request):
    """
    Jab bhi koi customer WhatsApp pe message bhejega, Meta ye
    endpoint call karega aur message ka poora data bhejega.
    Hum yaha se message nikaalte hain, AI se jawab lete hain,
    aur wapas customer ko bhej dete hain.
    """
    data = await request.json()
    print("📩 Naya message aaya:", data)

    try:
        # Meta ka data structure thoda nested hota hai,
        # isliye layer by layer andar jaake message nikaalte hain
        entry = data["entry"][0]
        changes = entry["changes"][0]
        value = changes["value"]

        # Agar koi actual message aaya hai (aur ye sirf status
        # update jaise "delivered/read" nahi hai)
        if "messages" in value:
            message_data = value["messages"][0]
            sender_number = message_data["from"]  # customer ka number
            user_text = message_data["text"]["body"]  # customer ne kya likha

            print(f"👤 {sender_number} ne likha: {user_text}")

            # AI se jawab lo (yehi function jo chat.py me test kiya tha)
            ai_answer = get_ai_response(user_text)
            print(f"🤖 AI ka jawab: {ai_answer}")

            # Jawab wapas WhatsApp pe bhejo
            send_whatsapp_message(sender_number, ai_answer)

    except (KeyError, IndexError) as e:
        # Agar ye status update tha (jaise "message delivered"),
        # to usme "messages" key nahi hoti - usse simply ignore karo
        print("Status update tha, message nahi. Ignore kar rahe hain.")

    return {"status": "ok"}


def send_whatsapp_message(to_number: str, message_text: str):
    """
    Meta ke Graph API ko call karke customer ko WhatsApp message bhejta hai.
    """
    url = f"https://graph.facebook.com/v21.0/{META_PHONE_NUMBER_ID}/messages"

    headers = {
        "Authorization": f"Bearer {META_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"body": message_text}
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        print("✅ Message safaltapoorvak bhej diya!")
    else:
        print(f"❌ Message bhejne me error: {response.text}")


@app.get("/")
async def home():
    """Bas ye check karne ke liye ki server chal raha hai ya nahi."""
    return {"message": "WhatsApp AI Bot chal raha hai! 🚀"}