import os
import requests
from datetime import datetime, timezone
from typing import List
from supabase import ClientOptions, create_client, Client
from schemas import Alert

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY, options=ClientOptions(schema="public"))

class NotificationService:
    def __init__(self):
        self.telegram_bot_token = TELEGRAM_BOT_TOKEN
        self.telegram_chat_id = TELEGRAM_CHAT_ID
        self.telegram_api_url = f"https://api.telegram.org/bot{self.telegram_bot_token}"
    
    def get_unnotified_alerts(self) -> List[Alert]:
        """Get all alerts that haven't been notified yet"""
        try:
            result = supabase.table("alerts").select("*").eq("notified", False).execute()
            return [Alert(**row) for row in result.data]
        except Exception as e:
            print(f"Error fetching unnotified alerts: {e}")
            return []
    
    def mark_alert_as_notified(self, alert_id: str) -> bool:
        """Mark alert as notified in database"""
        try:
            result = supabase.table("alerts").update({
                "notified": True
            }).eq("id", alert_id).execute()
            return bool(result.data)
        except Exception as e:
            print(f"Error marking alert as notified: {e}")
            return False
    
    def send_telegram_message(self, message: str) -> bool:
        """Send message via Telegram Bot API"""
        try:
            url = f"{self.telegram_api_url}/sendMessage"
            data = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            
            response = requests.post(url, data=data, timeout=10)
            response.raise_for_status()
            
            return response.json().get("ok", False)
        except Exception as e:
            print(f"Error sending Telegram message: {e}")
            return False
    
    def send_notification(self, alert: Alert) -> bool:
        """Send notification for an alert via Telegram"""
        try:
            # Format the alert message for Telegram
            direction_emoji = "📈" if alert.trigger_direction == "above" else "📉"
            message = f"""
🚨 <b>CRYPTO PRICE ALERT</b> 🚨

<b>Pair:</b> {alert.pair}
<b>Direction:</b> {direction_emoji} {alert.trigger_direction.upper()}
<b>Target Price:</b> ${alert.target_price:,.2f}
<b>Triggered Price:</b> ${alert.triggered_price:,.2f}
<b>Type:</b> {alert.trigger_type.title()}
<b>Time:</b> {alert.triggered_at.strftime('%Y-%m-%d %H:%M:%S UTC')}

<b>Alert ID:</b> {alert.id}
            """.strip()
            
            # Send via Telegram
            if self.send_telegram_message(message):
                print("✅ Telegram notification sent successfully")
                return True
            else:
                print("❌ Failed to send Telegram notification")
                return False
                
        except Exception as e:
            print(f"Error sending notification: {e}")
            return False
    
    def process_notifications(self) -> int:
        """Process all unnotified alerts"""
        alerts = self.get_unnotified_alerts()
        sent_count = 0
        
        print(f"Processing {len(alerts)} unnotified alerts...")
        
        for alert in alerts:
            print(f"Sending notification for alert {alert.id}...")
            
            if self.send_notification(alert):
                if self.mark_alert_as_notified(alert.id):
                    print(f"✅ Notification sent and marked for alert {alert.id}")
                    sent_count += 1
                else:
                    print(f"❌ Failed to mark alert {alert.id} as notified")
            else:
                print(f"❌ Failed to send notification for alert {alert.id}")
        
        return sent_count
    
    def run_notification_cycle(self) -> int:
        """Run one complete notification cycle"""
        print("📧 Starting notification cycle...")
        sent_count = self.process_notifications()
        
        if sent_count > 0:
            print(f"✅ {sent_count} notifications sent!")
        else:
            print("ℹ️  No notifications to send")
        
        return sent_count