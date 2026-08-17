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

from chat import get_ai_response

load_dotenv()

META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
META_PHONE_NUMBER_ID = os.getenv("META_PHONE_NUMBER_ID")

# Must match the verify token configured in the Meta app dashboard
VERIFY_TOKEN = "mera_secret_token_123"

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

            print(f"From {sender_number}: {user_text}")

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


@app.get("/")
async def home():
    """Health check endpoint."""
    return {"message": "WhatsApp AI Bot is running"}