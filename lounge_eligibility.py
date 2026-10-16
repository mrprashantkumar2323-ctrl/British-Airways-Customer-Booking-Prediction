"""
British Airways - Heathrow Terminal 3 Lounge Eligibility Checker
Author: Prashant Kumar
"""

def evaluate_lounge_eligibility(cabin_class, tier_status, flight_origin, flight_destination):
    """
    Evaluates passenger eligibility for British Airways Galleries & Concorde lounges at LHR T3.
    """
    eligible = False
    lounge_type = "None"
    
    tier_status = tier_status.capitalize()
    cabin_class = cabin_class.capitalize()

    if tier_status == "Gold" or cabin_class == "First":
        eligible = True
        lounge_type = "Galleries First / Concorde Room"
    elif tier_status == "Silver" or cabin_class in ["Club world", "Club europe", "Business"]:
        eligible = True
        lounge_type = "Galleries Club Lounge"
    elif tier_status in ["Bronze", "Blue"]:
        eligible = False
        lounge_type = "Not Eligible (Upgrade Required)"

    return {
        "Eligible": eligible,
        "Lounge Type": lounge_type,
        "Terminal": "Heathrow Terminal 3",
        "Cabin": cabin_class,
        "Frequent Flyer Tier": tier_status
    }

if __name__ == "__main__":
    passengers = [
        {"cabin_class": "Economy", "tier_status": "Gold"},
        {"cabin_class": "Club World", "tier_status": "Blue"},
        {"cabin_class": "Economy", "tier_status": "Blue"}
    ]
    print("=== Heathrow T3 Lounge Eligibility Evaluation ===")
    for p in passengers:
        result = evaluate_lounge_eligibility(p["cabin_class"], p["tier_status"], "LHR", "JFK")
        print(f"Pass: {p['cabin_class']} ({p['tier_status']}) -> {result['Lounge Type']}")
