#!/usr/bin/env python3
"""
Helper script to get your Telegram Chat ID
Run this script and send a message to your bot to get your chat ID
"""

import os
import requests
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_chat_id():
    """Get chat ID by polling for updates from the bot"""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    
    if not bot_token:
        print("❌ TELEGRAM_BOT_TOKEN not found in environment variables")
        print("Please add your bot token to .env file")
        return None
    
    print("🤖 Telegram Chat ID Helper")
    print("=" * 40)
    print("1. Make sure you've started a conversation with your bot")
    print("2. Send any message to your bot (e.g., /start)")
    print("3. This script will detect your chat ID")
    print("=" * 40)
    print("Waiting for messages... (Press Ctrl+C to stop)")
    
    try:
        last_update_id = 0
        
        while True:
            # Get updates from Telegram
            url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
            params = {"offset": last_update_id + 1, "timeout": 10}
            
            try:
                response = requests.get(url, params=params, timeout=15)
                response.raise_for_status()
                data = response.json()
                
                if data.get("ok") and data.get("result"):
                    for update in data["result"]:
                        if "message" in update:
                            message = update["message"]
                            chat_id = message["chat"]["id"]
                            username = message["from"].get("username", "Unknown")
                            first_name = message["from"].get("first_name", "Unknown")
                            
                            print(f"\n✅ Chat ID found!")
                            print(f"Chat ID: {chat_id}")
                            print(f"User: {first_name} (@{username})")
                            print(f"Message: {message.get('text', 'No text')}")
                            print(f"\nAdd this to your .env file:")
                            print(f"TELEGRAM_CHAT_ID={chat_id}")
                            
                            return chat_id
                        
                        last_update_id = update["update_id"]
                
                time.sleep(1)  # Wait 1 second before next poll
                
            except requests.exceptions.Timeout:
                print(".", end="", flush=True)  # Show activity
                continue
            except Exception as e:
                print(f"\n❌ Error: {e}")
                break
                
    except KeyboardInterrupt:
        print("\n\n👋 Stopped by user")
        return None

def main():
    """Main function"""
    chat_id = get_chat_id()
    
    if chat_id:
        print(f"\n🎉 Success! Your chat ID is: {chat_id}")
        print("Add this to your .env file and you're ready to go!")
    else:
        print("\n❌ Could not get chat ID. Please try again.")

if __name__ == "__main__":
    main()
