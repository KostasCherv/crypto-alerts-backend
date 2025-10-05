import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from price_monitor import PriceMonitor
from trend_analysis import TrendAnalyzer
from notifications import NotificationService

# Load environment variables
load_dotenv()

def main():
    """Main function to run the crypto price alert system"""
    print("🚀 Crypto Price Alert System Starting...")
    print("=" * 60)
    print(f"Started at: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)
    
    # Initialize services
    price_monitor = PriceMonitor()
    trend_analyzer = TrendAnalyzer()
    notification_service = NotificationService()
    
    try:
        # Step 1: Monitor prices and check for triggers
        print("\n📊 STEP 1: Price Monitoring")
        print("-" * 30)
        triggered_count = price_monitor.run_monitoring_cycle()
        
        # Step 2: Run trend analysis
        print("\n📈 STEP 2: Trend Analysis")
        print("-" * 30)
        analyzed_count = trend_analyzer.run_trend_analysis()
        
        # Step 3: Send notifications
        print("\n📧 STEP 3: Notifications")
        print("-" * 30)
        sent_count = notification_service.run_notification_cycle()
        
        # Summary
        print("\n" + "=" * 60)
        print("📋 SUMMARY")
        print("=" * 60)
        print(f"✅ Price alerts triggered: {triggered_count}")
        print(f"✅ Trend analyses completed: {analyzed_count}")
        print(f"✅ Notifications sent: {sent_count}")
        print(f"🏁 Completed at: {datetime.now(timezone.utc).isoformat()}")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
