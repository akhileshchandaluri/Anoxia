"""
Test ANOXIA Backend API Endpoints
"""
import requests
import json
import time

BASE_URL = "http://localhost:5000"

def test_endpoints():
    """Test all API endpoints."""
    
    print("="*70)
    print("TESTING ANOXIA BACKEND API")
    print("="*70)
    
    # Wait for API to start
    time.sleep(2)
    
    # Test 1: Health check
    print("\n1. Health Check")
    try:
        resp = requests.get(f"{BASE_URL}/api/health", timeout=5)
        print(f"   Status: {resp.status_code}")
        print(f"   Response: {json.dumps(resp.json(), indent=2)[:200]}")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test 2: Precursor conditions at Arabian Sea location
    print("\n2. Precursor Conditions (Arabian Sea - 15.5°N, 70°E)")
    try:
        resp = requests.get(f"{BASE_URL}/api/precursor-conditions/15.5/70.0", timeout=5)
        print(f"   Status: {resp.status_code}")
        data = resp.json()
        print(f"   Location: {data['location']}")
        print(f"   Hypoxia Prob: {data['precursor_conditions'].get('hypoxia_probability')}")
        print(f"   Nitrate: +{data['precursor_conditions'].get('nitrate_anomaly')}%")
        print(f"   Thermal: {data['precursor_conditions'].get('thermal_stratification')}")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test 3: Precursor conditions at Bay of Bengal location
    print("\n3. Precursor Conditions (Bay of Bengal - 14°N, 90°E)")
    try:
        resp = requests.get(f"{BASE_URL}/api/precursor-conditions/14.0/90.0", timeout=5)
        print(f"   Status: {resp.status_code}")
        data = resp.json()
        print(f"   Location: {data['location']}")
        print(f"   Hypoxia Prob: {data['precursor_conditions'].get('hypoxia_probability')}")
        print(f"   Nitrate: +{data['precursor_conditions'].get('nitrate_anomaly')}%")
        print(f"   Thermal: {data['precursor_conditions'].get('thermal_stratification')}")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test 4: Wind vectors
    print("\n4. Wind Vectors")
    try:
        resp = requests.get(f"{BASE_URL}/api/wind-vectors", timeout=5)
        print(f"   Status: {resp.status_code}")
        data = resp.json()
        print(f"   Total vectors: {len(data['wind_vectors'])}")
        if data['wind_vectors']:
            v = data['wind_vectors'][0]
            print(f"   Sample: Lat={v['lat']}, Lon={v['lon']}, Speed={v['speed']} m/s")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test 5: Ocean currents
    print("\n5. Ocean Currents")
    try:
        resp = requests.get(f"{BASE_URL}/api/ocean-currents", timeout=5)
        print(f"   Status: {resp.status_code}")
        data = resp.json()
        print(f"   Total vectors: {len(data['current_vectors'])}")
        if data['current_vectors']:
            v = data['current_vectors'][0]
            print(f"   Sample: Lat={v['lat']}, Lon={v['lon']}, Speed={v['speed']} cm/s")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test 6: Fertilizer runoff
    print("\n6. Fertilizer Runoff")
    try:
        resp = requests.get(f"{BASE_URL}/api/fertilizer-runoff", timeout=5)
        print(f"   Status: {resp.status_code}")
        data = resp.json()
        print(f"   Nitrate grid points: {len(data['nitrate_grid']['lats'])}")
        print(f"   Runoff sources: {len(data['sources'])}")
        for source in data['sources'][:2]:
            print(f"     - {source['name']}: {source['discharge']} discharge")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test 7: Dead zone markers
    print("\n7. Dead Zone Markers")
    try:
        resp = requests.get(f"{BASE_URL}/api/dead-zone-markers", timeout=5)
        print(f"   Status: {resp.status_code}")
        data = resp.json()
        print(f"   Total dead zones found: {len(data['dead_zones'])}")
        if data['dead_zones']:
            dz = data['dead_zones'][0]
            print(f"   Sample: Lat={dz['lat']}, Lon={dz['lon']}, P(hypoxia)={dz['p_hypoxia']}, Status={dz['status']}")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)


if __name__ == '__main__':
    test_endpoints()
