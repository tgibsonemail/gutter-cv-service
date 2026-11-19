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
import sys
import time
import threading
from dotenv import load_dotenv

load_dotenv()

# Footage engine (Nov 2026): address -> gutter footage, range and flags. Lives in ./footage.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "footage"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PropertyRequest(BaseModel):
    lat: float
    lng: float
    footprint_coordinates: list

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
    api_key = os.getenv('MAPTILER_API_KEY')
    n = 2.0 ** zoom
    xtile = int((lng + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(math.radians(lat)) + 
                 (1 / math.cos(math.radians(lat)))) / math.pi) / 2.0 * n)
    tiles = []
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
    if not tiles:
        return None
    
    # Get the actual tile size from the first tile (handles both 256x256 and 512x512)
    first_tile = tiles[0][2]
    tile_size = first_tile.shape[0]
    
    # Create canvas based on actual tile size
    canvas = np.zeros((tile_size * 3, tile_size * 3, 3), dtype=np.uint8)
    
    for dx, dy, tile_img in tiles:
        x_offset = (dx + 1) * tile_size
        y_offset = (dy + 1) * tile_size
        canvas[y_offset:y_offset+tile_size, x_offset:x_offset+tile_size] = tile_img
    
    return canvas

def calculate_perimeter(coordinates):
    def haversine(lat1, lon1, lat2, lon2):
        R = 20902231
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
    if image is None:
        return 0.85
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                minLineLength=30, maxLineGap=10)
        if lines is None or len(lines) < 4:
            return 0.85
        horizontal_lines = 0
        total_lines = len(lines)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(math.atan2(y2 - y1, x2 - x1) * 180 / math.pi)
            if angle < 20 or angle > 160:
                horizontal_lines += 1
        if total_lines > 0:
            eave_ratio = 0.75 + (horizontal_lines / total_lines) * 0.22
            return min(0.97, max(0.75, eave_ratio))
        return 0.85
    except Exception as e:
        print(f"CV analysis failed: {e}")
        return 0.85

def estimate_stories(footprint_coords, image=None):
    perimeter = calculate_perimeter(footprint_coords)
    if perimeter < 160:
        return 2
    elif perimeter > 220:
        return 1
    else:
        return 1

@app.post("/analyze", response_model=MeasurementResponse)
async def analyze_property(request: PropertyRequest):
    try:
        perimeter_ft = calculate_perimeter(request.footprint_coordinates)
        tiles = get_maptiler_tiles(request.lat, request.lng)
        image = stitch_tiles(tiles) if tiles else None
        eave_ratio = analyze_roof_edges(image, request.footprint_coordinates)
        stories = estimate_stories(request.footprint_coordinates, image)
        gutter_lf = perimeter_ft * 1.05 * eave_ratio  # 1.05: wall outlines (Overture) + ~5% eave overhang. Was 1.15 (quotes ran ~10% high vs measured, Nov 2026)
        downspout_count = max(4, min(10, math.ceil(gutter_lf / 35)))
        if stories >= 2:
            downspout_count += 1
        downspout_lf = downspout_count * (stories * 10 + 5)
        total_lf = gutter_lf + downspout_lf
        margin = 0.10
        gutter_lf_min = gutter_lf * (1 - margin)
        gutter_lf_max = gutter_lf * (1 + margin)
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

class FootageRequest(BaseModel):
    address: str | None = None
    lat: float | None = None
    lng: float | None = None


# One footage measurement at a time (Starter plan has 512 MB; one run peaks near 310 MB).
# A second request waits its turn; if it waits too long it gets status "busy" so the quote engine falls back.
# The quote engine gives up after 30 s, so waiting + measuring must stay under that.
_FOOTAGE_LOCK = threading.Lock()
_FOOTAGE_WAIT_S = 15
_TOTAL_BUDGET_S = 27     # waiting + measuring; the engine itself caps a run at 25 s


def _scrub(text, key):
    text = str(text)
    return text.replace(key, "***") if key else text


@app.post("/footage")
def footage(request: FootageRequest):
    # Never raises: any problem comes back as status "error" so the quote engine can fall back.
    key = (os.getenv("GOOGLE_MAPS_API_KEY") or "").strip()
    t0 = time.time()
    if not key:
        return {"status": "error", "message": "GOOGLE_MAPS_API_KEY not set on the measuring service", "footage_ft": None}
    if not request.address and (request.lat is None or request.lng is None):
        return {"status": "error", "message": "address or lat/lng required", "footage_ft": None}
    if not _FOOTAGE_LOCK.acquire(timeout=_FOOTAGE_WAIT_S):
        return {"status": "busy", "message": "another measurement was running; waited %ds" % _FOOTAGE_WAIT_S,
                "footage_ft": None, "seconds": round(time.time() - t0, 1)}
    try:
        from estimate import estimate
        waited = time.time() - t0
        result = estimate(address=request.address, lat=request.lat, lng=request.lng, key=key,
                          time_budget_s=max(8.0, min(25.0, _TOTAL_BUDGET_S - waited)))
    except BaseException as e:  # the engine uses sys.exit for some Google errors
        return {"status": "error", "message": _scrub(e, key)[:300], "footage_ft": None,
                "seconds": round(time.time() - t0, 1)}
    finally:
        _FOOTAGE_LOCK.release()
    result["queue_wait_s"] = round(time.time() - t0 - (result.get("timings_s") or {}).get("total", 0), 1) if isinstance(result.get("timings_s"), dict) else None
    if isinstance(result, dict) and result.get("message"):
        result["message"] = _scrub(result["message"], key)
    for f in (result.get("flags") or []):
        if isinstance(f, dict) and f.get("message"):
            f["message"] = _scrub(f["message"], key)
    return result


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "gutter-cv-analysis"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
