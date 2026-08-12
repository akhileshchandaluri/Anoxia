#!/usr/bin/env python
"""Quick diagnostic for erddapy issues."""

try:
    from erddapy import ERDDAP
    print("✓ erddapy imported successfully")
    print(f"  ERDDAP class: {ERDDAP}")
    print(f"  ERDDAP.__init__ signature:")
    import inspect
    sig = inspect.signature(ERDDAP.__init__)
    print(f"    {sig}")
    
    # Try to instantiate
    print("\nTrying to instantiate ERDDAP...")
    e = ERDDAP(
        server="https://coastwatch.pfeg.noaa.gov/erddap",
        protocol="tabledap",
        response="csv",
    )
    print("✓ ERDDAP instantiated successfully")
    
    # Now try setting dataset_id
    print("\nSetting dataset_id attribute...")
    e.dataset_id = "ArgoFloats"
    print(f"✓ dataset_id set to: {e.dataset_id}")
    
    # Try setting constraints
    print("\nSetting constraints...")
    e.constraints = {
        "longitude>=": 55,
        "longitude<=": 100,
        "latitude>=": 0,
        "latitude<=": 30,
    }
    print(f"✓ constraints set")
    
    # Try setting variables
    print("\nSetting variables...")
    e.variables = ["longitude", "latitude", "time", "pres", "doxy"]
    print(f"✓ variables set")
    
    # Try to get URL
    print("\nGetting URL...")
    url = e.get_download_url()
    print(f"✓ URL generated:\n  {url}")
    
except Exception as ex:
    import traceback
    print(f"✗ Error: {type(ex).__name__}: {ex}")
    traceback.print_exc()
