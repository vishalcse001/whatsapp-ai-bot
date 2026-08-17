"""
main.py

FastAPI server that receives WhatsApp messages via Meta's webhook,
generates a response using the RAG pipeline in chat.py, and sends
the reply back to the user through the WhatsApp Cloud API.
"""

import os
import requests
from fastapi import FastAPI, Request
from dotenv import load_dotenv

from chat import get_ai_response, get_welcome_message

load_dotenv()

META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
META_PHONE_NUMBER_ID = os.getenv("META_PHONE_NUMBER_ID")
GOOGLE_SHEET_WEBHOOK_URL = os.getenv("GOOGLE_SHEET_WEBHOOK_URL")

# Must match the verify token configured in the Meta app dashboard
VERIFY_TOKEN = "mera_secret_token_123"

# Common greetings that trigger a dynamic welcome message instead of
# going through the regular RAG pipeline.
GREETINGS = {"hi", "hello", "hey", "namaste", "hii", "hlo", "helo"}

app = FastAPI()


@app.get("/webhook")
async def verify_webhook(request: Request):
    """
    Handles Meta's one-time webhook verification handshake.
    Called automatically when the webhook URL is registered in the app dashboard.
    """
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        print("Webhook verified successfully.")
        return int(challenge)
    return {"error": "Verification failed"}


@app.post("/webhook")
async def receive_message(request: Request):
    """
    Handles incoming WhatsApp messages. Extracts the sender and message
    text from Meta's payload, generates a response via the RAG pipeline,
    and sends the reply back to the sender.
    """
    data = await request.json()
    print("Incoming payload:", data)

    try:
        entry = data["entry"][0]
        changes = entry["changes"][0]
        value = changes["value"]

        if "messages" in value:
            message_data = value["messages"][0]
            sender_number = message_data["from"]
            user_text = message_data["text"]["body"]

            # Extract the sender's display name, if WhatsApp provided one
            sender_name = "Unknown"
            try:
                sender_name = value["contacts"][0]["profile"]["name"]
            except (KeyError, IndexError):
                pass

            print(f"From {sender_number}: {user_text}")

            # Log every inbound message as a lead in the Google Sheet
            log_lead(sender_name, sender_number, user_text)

            # Greetings get a dynamically generated welcome message based on
            # the loaded document's own content, rather than a fixed reply.
            if user_text.strip().lower() in GREETINGS:
                ai_answer = get_welcome_message()
            else:
                ai_answer = get_ai_response(user_text)

            print(f"Response: {ai_answer}")

            send_whatsapp_message(sender_number, ai_answer)

    except (KeyError, IndexError):
        # Status update events (delivered/read) don't contain a "messages" key
        print("Received a status update, not a message. Ignoring.")

    return {"status": "ok"}


def send_whatsapp_message(to_number: str, message_text: str):
    """Sends a text message to a WhatsApp user via the Meta Graph API."""
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
        print("Message sent successfully.")
    else:
        print(f"Failed to send message: {response.text}")


def log_lead(name: str, phone: str, message: str):
    """
    Sends the customer's name, phone number, and message to the Google Sheet
    (via the Apps Script web app) so every inbound conversation is captured
    as a lead. Failures here are logged but never block the chat reply.
    """
    if not GOOGLE_SHEET_WEBHOOK_URL:
        return

    payload = {"name": name, "phone": phone, "message": message}

    try:
        response = requests.post(GOOGLE_SHEET_WEBHOOK_URL, json=payload, timeout=10)
        if response.status_code == 200:
            print("Lead logged to Google Sheet.")
        else:
            print(f"Failed to log lead: {response.text}")
    except requests.RequestException as e:
        print(f"Error logging lead: {e}")


@app.get("/")
async def home():
    """Health check endpoint."""
    return {"message": "WhatsApp AI Bot is running"}