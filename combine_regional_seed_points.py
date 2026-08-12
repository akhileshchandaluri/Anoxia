"""
Combine GFW seed points from Arabian Sea and Bay of Bengal regions.
"""

import json
from pathlib import Path


def combine_seed_points(arabian_sea_json: str, bay_of_bengal_json: str, 
                        output_json: str = "./data/gfw/gfw_seed_points.json") -> None:
    """
    Combine seed points from two regional JSON files into one.
    """
    
    # Load Arabian Sea points
    with open(arabian_sea_json, 'r') as f:
        arabian_data = json.load(f)
    
    # Load Bay of Bengal points
    with open(bay_of_bengal_json, 'r') as f:
        bengal_data = json.load(f)
    
    # Combine seed points
    combined_points = arabian_data['seed_points'] + bengal_data['seed_points']
    
    # Create combined output
    combined_data = {
        "seed_points": combined_points,
        "count": len(combined_points),
        "regions": {
            "Arabian_Sea": {
                "points": arabian_data['seed_points'],
                "count": len(arabian_data['seed_points']),
                "bbox": {
                    "lat_min": 5,
                    "lat_max": 25,
                    "lon_min": 55,
                    "lon_max": 75
                }
            },
            "Bay_of_Bengal": {
                "points": bengal_data['seed_points'],
                "count": len(bengal_data['seed_points']),
                "bbox": {
                    "lat_min": 5,
                    "lat_max": 25,
                    "lon_min": 80,
                    "lon_max": 95
                }
            }
        },
        "combined_bbox": {
            "description": "Indian Ocean Study Regions",
            "lat_min": 5,
            "lat_max": 25,
            "lon_min": 55,
            "lon_max": 95
        }
    }
    
    # Save combined output
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(combined_data, f, indent=2)
    
    print(f"Combined {len(combined_points)} seed points:")
    print(f"  - Arabian Sea: {len(arabian_data['seed_points'])} points")
    print(f"  - Bay of Bengal: {len(bengal_data['seed_points'])} points")
    print(f"  Saved to: {output_json}")


if __name__ == "__main__":
    combine_seed_points(
        "./data/gfw/gfw_arabian_sea.json",
        "./data/gfw/gfw_bay_of_bengal.json"
    )
