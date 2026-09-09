"""
Intent definitions and few-shot examples for the AI Customer Support Agent.
"""

from typing import Dict, List

INTENT_DEFINITIONS: Dict[str, str] = {
    "device_troubleshoot": "iPhone/iPad/Mac not working, restart, update, battery, screen issues",
    "account_access": "Apple ID, password reset, two-factor auth, locked account",
    "billing_payment": "charges, refunds, subscription, payment method issues",
    "warranty_repair": "device damage, AppleCare, repair status, replacement",
    "app_issue": "App Store, app crashes, download problems, app updates",
    "connectivity": "WiFi, Bluetooth, cellular, AirDrop issues",
    "data_recovery": "backup, iCloud storage, data transfer, restore",
    "general_inquiry": "product info, how-to questions, feature questions",
    "feedback_complaint": "dissatisfaction, escalation requests, poor experience",
    "other": "doesn't fit above categories"
}

def get_intent_description(intent: str) -> str:
    """Get the description for a specific intent."""
    return INTENT_DEFINITIONS.get(intent, "Unknown intent")

def get_few_shot_examples() -> Dict[str, List[str]]:
    """Get two realistic Apple Support customer tweets per intent."""
    return {
        "device_troubleshoot": [
            "@AppleSupport my iPhone 13 Pro is completely frozen and won't turn on despite holding the power button.",
            "Battery drain on iOS 16.4 is insane... losing 20% in an hour just browsing Safari. Pls fix @AppleSupport!"
        ],
        "account_access": [
            "@AppleSupport I forgot my Apple ID password and the reset email is going to an old email address I can't access.",
            "My account has been locked for security reasons and I can't verify my two-factor auth because I lost my phone!"
        ],
        "billing_payment": [
            "@AppleSupport why did I just get charged $9.99 for Apple Music when I canceled my subscription last week?",
            "Trying to update my payment method but it keeps saying my card is declined even though it works everywhere else."
        ],
        "warranty_repair": [
            "Just dropped my iPad and the screen is shattered. Does AppleCare+ cover this? @AppleSupport",
            "@AppleSupport checking the repair status of my Mac, the tracking hasn't updated in 3 days."
        ],
        "app_issue": [
            "@AppleSupport the App Store won't let me download any apps, it just spins forever.",
            "Instagram keeps crashing immediately upon opening after the latest iOS update. Anyone else?"
        ],
        "connectivity": [
            "@AppleSupport my AirPods won't connect via Bluetooth anymore, I've tried resetting them 5 times.",
            "WiFi keeps dropping on my M1 MacBook Pro but all other devices in the house are fine."
        ],
        "data_recovery": [
            "@AppleSupport my iCloud backup is stuck at 99% for the past two days, how do I fix this?",
            "Upgraded my phone but the WhatsApp chat history didn't transfer over. Please tell me my data isn't gone."
        ],
        "general_inquiry": [
            "@AppleSupport does the new Apple Watch Series 8 have a blood oxygen sensor?",
            "How do I take a screenshot on an iPhone 14 Pro? The buttons changed."
        ],
        "feedback_complaint": [
            "@AppleSupport worst customer service ever at the local store today. Waited 2 hours past my appointment time.",
            "Absolutely ridiculous that they removed the headphone jack. Thinking of switching to Android."
        ],
        "other": [
            "@AppleSupport look at this cool photo I took with my new iPhone!",
            "When is the next Apple event happening?"
        ]
    }
