from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import math
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Allow CORS from Lovable
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PropertyRequest(BaseModel):
    lat: float
    lng: float
    footprint_coordinates: list  # [[lng, lat], [lng, lat], ...]

class MeasurementResponse(BaseModel):
    gutter_linear_feet: float
    gutter_linear_feet_min: float
    gutter_linear_feet_max: float
    downspout_count: int
    downspout_linear_feet: float
    total_linear_feet: float
    stories: int
    confidence: str
    eave_ratio: float
    building_perimeter_feet: float

def get_maptiler_tiles(lat, lng, zoom=20):
    """Fetch satellite imagery tiles from MapTiler"""
    api_key = os.getenv('MAPTILER_API_KEY')
    
    # Calculate tile coordinates
    n = 2.0 ** zoom
    xtile = int((lng + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(math.radians(lat)) + 
                 (1 / math.cos(math.radians(lat)))) / math.pi) / 2.0 * n)
    
    tiles = []
    # Fetch a 3x3 grid of tiles around the property
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            tile_url = f"https://api.maptiler.com/maps/satellite/{zoom}/{xtile+dx}/{ytile+dy}.jpg?key={api_key}"
            try:
                response = requests.get(tile_url, timeout=10)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content))
                    tiles.append((dx, dy, np.array(img)))
            except Exception as e:
                print(f"Failed to fetch tile ({dx}, {dy}): {e}")
    
    return tiles

def stitch_tiles(tiles):
    """Stitch tiles into single image"""
    if not tiles:
        return None
    
    # Assuming 256x256 tiles
    tile_size = 256
    
    # Create canvas for 3x3 grid
    canvas = np.zeros((tile_size * 3, tile_size * 3, 3), dtype=np.uint8)
    
    for dx, dy, tile_img in tiles:
        x_offset = (dx + 1) * tile_size
        y_offset = (dy + 1) * tile_size
        canvas[y_offset:y_offset+tile_size, x_offset:x_offset+tile_size] = tile_img
    
    return canvas

def calculate_perimeter(coordinates):
    """Calculate perimeter from lat/lng coordinates in feet"""
    # Haversine formula for distance between points
    def haversine(lat1, lon1, lat2, lon2):
        R = 20902231  # Earth radius in feet
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        return R * c
    
    perimeter = 0
    for i in range(len(coordinates)):
        lon1, lat1 = coordinates[i]
        lon2, lat2 = coordinates[(i + 1) % len(coordinates)]
        perimeter += haversine(lat1, lon1, lat2, lon2)
    
    return perimeter

def analyze_roof_edges(image, footprint_coords):
    """Use CV to estimate what % of perimeter has gutters (eaves vs rakes)"""
    if image is None:
        return 0.85  # Conservative default
    
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Hough line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                minLineLength=30, maxLineGap=10)
        
        if lines is None or len(lines) < 4:
            return 0.85  # Default if can't detect enough lines
        
        # Analyze line directions
        horizontal_lines = 0
        total_lines = len(lines)
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(math.atan2(y2 - y1, x2 - x1) * 180 / math.pi)
            # Consider lines within 20° of horizontal as potential eaves
            if angle < 20 or angle > 160:
                horizontal_lines += 1
        
        # Eave ratio based on horizontal line dominance
        if total_lines > 0:
            eave_ratio = 0.75 + (horizontal_lines / total_lines) * 0.22
            return min(0.97, max(0.75, eave_ratio))
        
        return 0.85
    
    except Exception as e:
        print(f"CV analysis failed: {e}")
        return 0.85  # Safe default

def estimate_stories(footprint_coords, image=None):
    """Estimate number of stories (1 or 2)"""
    # For now, use simple heuristic based on footprint size
    # Could be enhanced with shadow analysis from imagery
    
    perimeter = calculate_perimeter(footprint_coords)
    
    # Rough heuristic: smaller footprint often means 2 stories
    # This is a placeholder - shadow analysis would be better
    if perimeter < 160:
        return 2
    elif perimeter > 220:
        return 1
    else:
        return 1  # Default to 1 story (user will correct if needed)

@app.post("/analyze", response_model=MeasurementResponse)
async def analyze_property(request: PropertyRequest):
    """Main endpoint: analyze property and return gutter measurements"""
    
    try:
        # 1. Calculate building perimeter
        perimeter_ft = calculate_perimeter(request.footprint_coordinates)
        
        # 2. Fetch satellite imagery
        tiles = get_maptiler_tiles(request.lat, request.lng)
        image = stitch_tiles(tiles) if tiles else None
        
        # 3. Run CV analysis to get eave ratio
        eave_ratio = analyze_roof_edges(image, request.footprint_coordinates)
        
        # 4. Estimate stories
        stories = estimate_stories(request.footprint_coordinates, image)
        
        # 5. Calculate gutter linear feet
        # Formula: Perimeter × overhang factor × eave ratio
        gutter_lf = perimeter_ft * 1.08 * eave_ratio
        
        # 6. Calculate downspouts
        downspout_count = max(4, min(10, math.ceil(gutter_lf / 35)))
        if stories >= 2:
            downspout_count += 1
        
        downspout_lf = downspout_count * (stories * 10 + 5)
        
        # 7. Total linear feet
        total_lf = gutter_lf + downspout_lf
        
        # 8. Calculate range (±10%)
        margin = 0.10
        gutter_lf_min = gutter_lf * (1 - margin)
        gutter_lf_max = gutter_lf * (1 + margin)
        
        # 9. Determine confidence
        if image is not None and eave_ratio > 0.80 and eave_ratio < 0.95:
            confidence = "High"
        elif eave_ratio == 0.85:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        return MeasurementResponse(
            gutter_linear_feet=round(gutter_lf, 1),
            gutter_linear_feet_min=round(gutter_lf_min, 1),
            gutter_linear_feet_max=round(gutter_lf_max, 1),
            downspout_count=downspout_count,
            downspout_linear_feet=round(downspout_lf, 1),
            total_linear_feet=round(total_lf, 1),
            stories=stories,
            confidence=confidence,
            eave_ratio=round(eave_ratio, 3),
            building_perimeter_feet=round(perimeter_ft, 1)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "gutter-cv-analysis"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

### D. Create .env File

**.env:**
MAPTILER_API_KEY=your_maptiler_key_here
