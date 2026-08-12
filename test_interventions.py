"""
Test script for ANOXIA Dynamic Intervention Recommendation System

Tests the new /api/intervention-measures/<lat>/<lon> endpoint with various scenarios.
"""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:5000"

# Test locations with expected characteristics
TEST_LOCATIONS = [
    {
        "name": "Bay of Bengal (High Risk)",
        "lat": 15.5,
        "lon": 88.0,
        "description": "High hypoxia probability zone"
    },
    {
        "name": "Arabian Sea (Coastal)",
        "lat": 18.5,
        "lon": 71.0,
        "description": "Agricultural runoff influence"
    },
    {
        "name": "Open Ocean (Low Risk)",
        "lat": 10.0,
        "lon": 75.0,
        "description": "Low precursor conditions"
    },
    {
        "name": "Godavari Delta",
        "lat": 16.7,
        "lon": 82.3,
        "description": "River runoff source"
    }
]

def print_header(text):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")

def test_precursor_conditions():
    """Test precursor conditions endpoint."""
    print_header("TEST 1: Precursor Conditions Endpoint")
    
    location = TEST_LOCATIONS[0]
    url = f"{BASE_URL}/api/precursor-conditions/{location['lat']}/{location['lon']}"
    
    print(f"📍 Location: {location['name']}")
    print(f"   Coordinates: ({location['lat']}, {location['lon']})")
    print(f"   Request: GET {url}\n")
    
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        print("✅ Response received:")
        print(json.dumps(data, indent=2))
        
        return data
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_interventions():
    """Test intervention measures endpoint."""
    print_header("TEST 2: Intervention Measures Endpoint")
    
    results = []
    
    for location in TEST_LOCATIONS:
        print(f"\n📍 Location: {location['name']}")
        print(f"   Coordinates: ({location['lat']}, {location['lon']})")
        print(f"   {location['description']}\n")
        
        url = f"{BASE_URL}/api/intervention-measures/{location['lat']}/{location['lon']}"
        
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            # Extract key info
            severity = data.get('severity_level', 'UNKNOWN')
            score = data.get('severity_score', 0)
            count = data.get('recommended_actions_count', 0)
            
            print(f"   Severity Level: {severity} (score: {score})")
            print(f"   Recommended Actions: {count}")
            
            # Show first 2 interventions
            interventions = data.get('interventions', [])
            if interventions:
                print(f"   Top Interventions:")
                for i, interv in enumerate(interventions[:2]):
                    print(f"      {i+1}. [{interv.get('priority', 'N/A')}] {interv.get('title', 'N/A')}")
                    print(f"         Reason: {interv.get('reason', 'N/A')}")
            
            results.append({
                'location': location['name'],
                'severity': severity,
                'score': score,
                'interventions': count
            })
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return results

def display_intervention_example():
    """Display a formatted example intervention."""
    print_header("EXAMPLE: Intervention Response Structure")
    
    example = {
        "location": {
            "lat": 15.5,
            "lon": 88.0
        },
        "zone": "Bay of Bengal Coastal",
        "precursor_summary": {
            "nitrate_anomaly": 180.5,
            "hypoxia_probability": 0.75,
            "thermal_stratification": "78%",
            "wind_stress": "15%",
            "do_drawdown_rate": "68%"
        },
        "severity_score": 2.45,
        "severity_level": "CRITICAL",
        "interventions": [
            {
                "title": "🚨 Emergency oxygen restoration deployment",
                "priority": "CRITICAL",
                "category": "Oxygen Restoration",
                "reason": "Very fast dissolved oxygen depletion rate detected (68%)",
                "impact": "Prevents complete anoxia and protects aquatic ecosystems",
                "timeline": "IMMEDIATE - within 48 hours",
                "measures": [
                    "Emergency oxygenation deployment",
                    "Increase freshwater flushing",
                    "Reduce hypoxic zone through nutrient control",
                    "Deploy oxygen diffusers in affected areas",
                    "🚨 Alert fisheries and maritime authorities"
                ]
            },
            {
                "title": "Reduce agricultural fertilizer runoff",
                "priority": "CRITICAL",
                "category": "Nutrient Reduction",
                "reason": "High nitrate anomaly detected (180.5% above baseline)",
                "impact": "Prevents algal bloom proliferation and oxygen depletion",
                "timeline": "Immediate - 2 months",
                "measures": [
                    "Implement precision agriculture with drip irrigation",
                    "Establish riparian buffer zones (500m minimum)",
                    "Reduce synthetic fertilizer use by 50-70%",
                    "Enforce wastewater treatment standards",
                    "Monitor discharge from agricultural runoff"
                ]
            },
            {
                "title": "⚠️ Regional coordination: Bay of Bengal",
                "priority": "HIGH",
                "category": "Regional Management",
                "reason": "Highest-risk zone for hypoxia development during monsoon",
                "impact": "Coordinates multi-stakeholder response in Bay of Bengal",
                "timeline": "Seasonal - pre/post monsoon",
                "measures": [
                    "Coordinate with Godavari/Mahanadi river dam operators",
                    "Manage monsoon surge planning",
                    "Monitor fishery impact zones",
                    "Implement regional nutrient policy"
                ]
            }
        ],
        "recommended_actions_count": 3,
        "metadata": {
            "timestamp": "2026-04-02T15:30:45.123456",
            "data_source": "dynamic_analysis",
            "ai_generated": True,
            "note": "AI-generated recommendations based on environmental conditions"
        }
    }
    
    print("Full Response Structure:")
    print(json.dumps(example, indent=2))

def show_severity_logic():
    """Show severity scoring logic."""
    print_header("SEVERITY SCORING LOGIC")
    
    print("Severity Score Calculation:")
    print("  • Nitrate anomaly (0-1.0):    >200% = 1.0, >150% = 0.8, >100% = 0.6, >50% = 0.4")
    print("  • DO Drawdown (0-1.0):        >70% = 1.0, >50% = 0.7, >30% = 0.4")
    print("  • Stagnation Risk (0-0.5):    High stratification + Low wind = 0.5")
    print("  • Hypoxia Probability (0-0.7): 70% weight on raw probability")
    print()
    print("Severity Levels:")
    print("  ✅ SAFE:     score < 1.0")
    print("  ⚠️  WARNING:  1.0 ≤ score < 2.0")
    print("  🚨 CRITICAL: score ≥ 2.0")

def main():
    print("\n" + "="*70)
    print("  ANOXIA - DYNAMIC INTERVENTION RECOMMENDATION SYSTEM")
    print("  Test Suite for /api/intervention-measures endpoint")
    print("="*70)
    
    # Check backend connectivity
    print("\n🔍 Checking backend connectivity...")
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=2)
        print("✅ Backend is running")
    except:
        print("❌ Backend is not running. Start with: python backend_api.py")
        return
    
    # Run tests
    test_precursor_conditions()
    results = test_interventions()
    display_intervention_example()
    show_severity_logic()
    
    # Summary
    print_header("TEST SUMMARY")
    print("✅ All tests completed!")
    print("\nTesting Results:")
    for result in results:
        print(f"  • {result['location']:30} | Severity: {result['severity']:10} | Actions: {result['interventions']}")
    
    print("\n📊 Frontend Integration:")
    print("  1. Open frontend: http://localhost:8050")
    print("  2. Click on map to fetch dynamic interventions")
    print("  3. Interventions UI cards will display immediately")
    print("  4. Try different locations (high/low risk zones)")

if __name__ == "__main__":
    main()
