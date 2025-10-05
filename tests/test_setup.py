#!/usr/bin/env python3
"""
Test script to verify the crypto alerts backend setup
"""

import os
from dotenv import load_dotenv
from supabase import ClientOptions, create_client, Client

def test_environment():
    """Test environment variables"""
    print("🔍 Testing environment variables...")
    
    load_dotenv()
    
    required_vars = ["SUPABASE_URL", "SUPABASE_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        return False
    
    print("✅ All required environment variables are set")
    return True

def test_supabase_connection():
    """Test Supabase connection"""
    print("\n🔍 Testing Supabase connection...")
    
    try:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        supabase = create_client(supabase_url, supabase_key, options=ClientOptions(schema="public"))
        
        # Test connection by querying a simple table
        result = supabase.table("price_levels").select("id").limit(1).execute()
        print("✅ Supabase connection successful")
        return True
        
    except Exception as e:
        print(f"❌ Supabase connection failed: {e}")
        return False

def test_binance_api():
    """Test Binance API connection"""
    print("\n🔍 Testing Binance API...")
    
    try:
        import requests
        
        url = "https://api.binance.com/api/v3/ticker/price"
        params = {"symbol": "BTCUSDT"}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ Binance API connection successful - BTC price: ${data['price']}")
        return True
        
    except Exception as e:
        print(f"❌ Binance API connection failed: {e}")
        return False

def test_telegram_api():
    """Test Telegram Bot API connection"""
    print("\n🔍 Testing Telegram Bot API...")
    
    try:
        import requests
        
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        url = f"https://api.telegram.org/bot{bot_token}/getMe"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if data.get("ok"):
            bot_info = data.get("result", {})
            print(f"✅ Telegram Bot API connection successful - Bot: @{bot_info.get('username', 'Unknown')}")
            
            # Test sending a message
            test_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            test_data = {
                "chat_id": chat_id,
                "text": "🤖 Crypto Alerts Bot is working!",
                "parse_mode": "HTML"
            }
            
            test_response = requests.post(test_url, data=test_data, timeout=10)
            test_response.raise_for_status()
            
            if test_response.json().get("ok"):
                print("✅ Test message sent successfully to Telegram")
                return True
            else:
                print("❌ Failed to send test message to Telegram")
                return False
        else:
            print("❌ Telegram Bot API connection failed")
            return False
        
    except Exception as e:
        print(f"❌ Telegram Bot API connection failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Crypto Alerts Backend - Setup Test")
    print("=" * 50)
    
    tests = [
        test_environment,
        test_supabase_connection,
        test_binance_api,
        test_telegram_api
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("✅ All tests passed! System is ready to use.")
    else:
        print("❌ Some tests failed. Please check the configuration.")
    
    return passed == total

if __name__ == "__main__":
    main()
