#!/usr/bin/env python3
"""
Gutter footage, version 2: plane based rules (agreed with Trey 2026-11-17).

Rules
 1. Split the roof into sloped planes. Every plane has one low edge. That is a candidate gutter run,
    whether it faces the yard or another building.
 2. Group planes into structures. The largest is the house. Other pitched structures are separate line items.
 3. Patio covers and carports (flatter than 2 in 12) are left off and treated as if they are not there.
    A house eave with a cover attached below it still counts.
 4. An upper roof that drains onto a lower roof of the same structure is left off.
 5. A low edge that meets another plane at the same height (a valley) is not a gutter run.
 6. Runs under 3 ft are dropped.
 7. Low edges hidden under a tree are counted and flagged as inferred.

Usage:
  python3 eave_planes.py --name house1 --address "123 Main St, Dallas, TX"
  python3 eave_planes.py --name house1 --lat 32.84 --lng -96.78
"""
import argparse, json, math, os, sys, urllib.parse
import numpy as np
from scipy import ndimage as ndi
from skimage import morphology, measure, draw
from PIL import Image, ImageDraw
from pyproj import Transformer
import netlimits as NL
import eave_test as E

M2FT = 3.28084
HERE = os.path.dirname(os.path.abspath(__file__))

COVER_PITCH_DEG = math.degrees(math.atan(2 / 12.0))  # 2 in 12 = 9.46 deg
MIN_ROOF_HEIGHT_M = 1.5    # anything lower than this above the ground is not roof
STEP_M = 1.2               # height jump inside the roof that marks a wall or upper story edge
DROP_ONTO_LOWER_M = 1.0    # rule 4: low edge this far above the next roof = drains onto lower roof
TREE_RISE_M = 0.6          # outside surface this much higher than the roof edge = tree
MIN_PLANE_M2 = 3.0
MIN_COVER_M2 = 4.0
MIN_RUN_FT = 3.0
RUN_GAP_M = 0.6            # a gap this long splits a run in two
WALK_PX = 25               # how far past a plane's low edge to look for what comes next (0.1 m each)
EXTEND_MAX_M = 1.2         # how far a run may be extended along the edge to reach the roof corner
END_ADD_M = 0.2            # added to each run for the blurred strip at the roof edge (to be calibrated)
EAVE_COS_O = 0.6           # outline pass: plane drains toward the edge
GABLE_COS_O = 0.4          # outline pass: plane drains along the edge
MAX_TREE_GAP_M = 30.0      # longest tree gap (measured around its square corners) that will be bridged
USE_TREE_REPAIR = False    # squaring off tree clipped corners is not reliable yet, so it is off
JOG_MIN_PX = 5             # the roof edge must step at least 0.5 m to count as a jog
RIDGE_GAP_PX = 11          # two planes within about 2 m of each other can share a ridge
RIDGE_MIN_PLANE_M2 = 4.0   # roof planes smaller than this are not used for ridge following
RIDGE_SEARCH_M = 6.0       # how far past the end of a ridge to look for the roof edge
RIDGE_DROP_MAX_M = 1.0     # roof within this of ridge height right at the edge = gable, lower = hip
RIDGE_EDGE_MAX_M = 4.0     # ridge ending within this distance of the roof edge = gable
GABLE_MIN_FT = 8.0
GABLE_MAX_FT = 70.0
USE_EDGE_HEIGHT = True     # decide gable vs eave from the roof edge height profile
EDGE_MAX_INSET_PX = 14     # roof mask starting more than 1.4 m inside the outline means the edge is hidden
GABLE_MIN_WIDTH_M = 3.0    # narrower than about 10 ft is not treated as a gable end
GABLE_MIN_RISE_M = 0.9     # the edge must climb at least about 3 ft
RAKE_SLOPE = 0.2           # edge climbing faster than 0.2 ft per ft is a gable edge
MIN_GABLE_FT = 6.0         # a gable stretch shorter than this between gutter runs is treated as gutter
INTERIOR_MIN_M = 1.5       # a plane run this far inside the outline is an interior run
STRUCT_CONTACT_M = 3.0     # planes must touch along at least this length to be one structure
PEAK_SEP_DEG = 25
PEAK_TOL_DEG = 30


def building_insights(lat, lng, key):
    return E.get_json("https://solar.googleapis.com/v1/buildingInsights:findClosest?" +
                      urllib.parse.urlencode({"location.latitude": lat, "location.longitude": lng,
                                              "requiredQuality": "MEDIUM", "key": key}))


def fetch(lat, lng, key, bi=None):
    if bi is None:
        bi = building_insights(lat, lng, key)
    c, bb = bi["center"], bi["boundingBox"]
    clat, clng = c["latitude"], c["longitude"]
    dlat = (bb["ne"]["latitude"] - bb["sw"]["latitude"]) * 111320.0
    dlng = (bb["ne"]["longitude"] - bb["sw"]["longitude"]) * 111320.0 * math.cos(math.radians(clat))
    radius = min(max(0.5 * math.hypot(dlat, dlng) + 8.0, 20.0), 75.0)
    for attempt in range(3):
        dl = E.get_json("https://solar.googleapis.com/v1/dataLayers:get?" +
                        urllib.parse.urlencode({"location.latitude": clat, "location.longitude": clng,
                                                "radiusMeters": round(radius, 1), "view": "IMAGERY_LAYERS",
                                                "requiredQuality": "MEDIUM", "pixelSizeMeters": 0.1,
                                                "key": key}))
        dsm, tags = E.get_tif(dl["dsmUrl"], key)
        mask, _ = E.get_tif(dl["maskUrl"], key)
        rgb, _ = E.get_tif(dl["rgbUrl"], key)
        rgb = rgb[..., :3].astype(np.uint8)
        dsm = dsm.astype(float)
        dsm[dsm < -1000] = np.nan
        mask = mask.astype(bool)
        mt = tags["ModelTransformationTag"]
        px, x0, y0 = float(mt[0]), float(mt[3]), float(mt[7])
        epsg = E.epsg_from_tags(tags)
        H, W = dsm.shape
        tr = Transformer.from_crs(4326, epsg, always_xy=True)
        x, y = tr.transform(clng, clat)
        cr, cc = int(round(min(max((y0 - y) / px, 0), H - 1))), int(round(min(max((x - x0) / px, 0), W - 1)))
        lab, nlab = ndi.label(mask)
        if nlab == 0:
            sys.exit("No roof found")
        if lab[cr, cc] > 0:
            target = lab[cr, cc]
        else:
            _, idx = ndi.distance_transform_edt(lab == 0, return_indices=True)
            target = lab[idx[0][cr, cc], idx[1][cr, cc]]
        blob = ndi.binary_fill_holes(lab == target)
        rr, cc_ = np.nonzero(blob)
        touches = bool(rr.min() <= 4 or cc_.min() <= 4 or rr.max() >= H - 5 or cc_.max() >= W - 5)
        if not touches or radius >= 99:
            break
        radius = min(radius * 1.6, 100.0)
    return dict(dsm=dsm, mask=mask, rgb=rgb, px=px, blob=blob, center=(cr, cc), x0=x0, y0=y0, epsg=epsg, clat=clat, clng=clng,
                quality=dl.get("imageryQuality"), date=dl.get("imageryDate"), touches=touches)


def repaired_polygon(blob, dsm, px):
    """Roof outline with tree covered edges removed and the visible edges extended to meet.
    A real roof edge has lower ground outside it. If the surface just outside is higher than the
    roof, a tree is covering the roof there."""
    H, W = blob.shape
    fin = np.isfinite(dsm)
    zs0 = E.norm_gauss(np.nan_to_num(dsm), blob & fin, 2.0)
    cons = measure.find_contours(np.pad(blob.astype(float), 1), 0.5)
    poly = measure.approximate_polygon(max(cons, key=len) - 1.0, tolerance=0.45 / px)
    if np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]
    N = len(poly)
    if N < 4:
        return None, 0
    lens = [float(np.hypot(*(poly[(i + 1) % N] - poly[i]))) for i in range(N)]
    il = int(np.argmax(lens))
    tl = (poly[(il + 1) % N] - poly[il]) / lens[il]
    tp = (poly[il] + poly[(il + 1) % N]) / 2 + 4 * np.array([tl[1], -tl[0]])
    ri, ci = int(round(tp[0])), int(round(tp[1]))
    sgn = -1.0 if (0 <= ri < H and 0 <= ci < W and blob[ri, ci]) else 1.0

    def z_at(arr, q):
        ri, ci = int(round(q[0])), int(round(q[1]))
        if 0 <= ri < H and 0 <= ci < W and np.isfinite(arr[ri, ci]):
            return float(arr[ri, ci])
        return None
    occ = []
    for i in range(N):
        p0, p1, L = poly[i], poly[(i + 1) % N], lens[i]
        if L < 1e-6:
            occ.append(False)
            continue
        t = (p1 - p0) / L
        n = np.array([t[1], -t[0]]) * sgn
        k = max(1, int(round(L * px / 0.2)))
        hits = tot = 0
        for j in range(k):
            p = p0 + t * L * (j + 0.5) / k
            zi = z_at(zs0, p - n * (0.6 / px))
            zo = [z_at(dsm, p + n * (dm / px)) for dm in (0.5, 0.9, 1.3)]
            zo = [z for z in zo if z is not None]
            if zi is None or not zo:
                continue
            tot += 1
            hits += int(max(zo) > zi + TREE_RISE_M)
        occ.append(tot > 0 and hits / tot > 0.3)
    if not any(occ) or all(occ):
        return None, 0
    vis = [i for i in range(N) if not occ[i]]
    st = {i: poly[i].copy() for i in vis}
    en = {i: poly[(i + 1) % N].copy() for i in vis}
    reps = 0
    for a_i, i in enumerate(vis):
        j = vis[(a_i + 1) % len(vis)]
        if j == (i + 1) % N or i == j:
            continue
        reps += 1
        A0, A1, B0, B1 = st[i], en[i], st[j], en[j]
        da, db = A1 - A0, B1 - B0
        den = da[0] * db[1] - da[1] * db[0]
        cosang = abs(float(np.dot(da, db))) / (np.hypot(*da) * np.hypot(*db) + 1e-9)
        if abs(den) < 1e-9 or cosang > math.cos(math.radians(20)):
            continue
        w = B0 - A0
        ta = (w[0] * db[1] - w[1] * db[0]) / den
        tb = (w[0] * da[1] - w[1] * da[0]) / den
        X = A0 + ta * da
        lim = 12.0 / px
        if ta > 0.5 and tb < 0.5 and np.hypot(*(X - A1)) < lim and np.hypot(*(X - B0)) < lim:
            en[i] = X
            st[j] = X
    pts = []
    for i in vis:
        for q in (st[i], en[i]):
            if not pts or np.hypot(*(q - pts[-1])) > 1.0:
                pts.append(q)
    if len(pts) < 3:
        return None, 0
    return np.array(pts), reps


def fema_footprint(lat, lng, D):
    """Open building footprint (FEMA / Oak Ridge USA Structures) for this house, in picture coordinates,
    nudged to line up with Google's roof."""
    import urllib.request
    url = "https://services2.arcgis.com/FiaPA4ga0iQKduv3/arcgis/rest/services/USA_Structures_View/FeatureServer/0/query"
    q = {"geometry": "%f,%f" % (lng, lat), "geometryType": "esriGeometryPoint", "inSR": "4326",
         "spatialRel": "esriSpatialRelIntersects", "distance": "15", "units": "esriSRUnit_Meter",
         "outFields": "*", "outSR": "4326", "f": "geojson", "resultRecordCount": "8"}
    d = json.loads(NL.read(url + "?" + urllib.parse.urlencode(q)))
    tr = Transformer.from_crs(4326, D["epsg"], always_xy=True)
    px, blob = D["px"], D["blob"]
    H, W = blob.shape
    best = None
    for f in d.get("features", []):
        g = f["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        for p in polys:
            ring = p[0][:-1]
            xs, ys = tr.transform([c[0] for c in ring], [c[1] for c in ring])
            rc = np.c_[(D["y0"] - np.array(ys)) / px, (np.array(xs) - D["x0"]) / px]
            m = draw.polygon2mask((H, W), rc)
            ov = (m & blob).sum()
            if best is None or ov > best[0]:
                best = (ov, rc, f["properties"])
    if best is None or best[0] == 0:
        return None, None
    rc = best[1]
    # nudge: different imagery sources sit a meter or two apart
    top = (-1, 0, 0)
    for dr in range(-40, 41, 4):
        for dc in range(-40, 41, 4):
            m = draw.polygon2mask((H, W), rc + [dr, dc])
            inter = (m & blob).sum()
            score = inter / max(blob.sum(), 1) - 0.5 * ((m & ~blob).sum() == 0)
            # favour covering Google's whole visible roof
            if inter > top[0]:
                top = (inter, dr, dc)
    return rc + [top[1], top[2]], dict(best[2], shift_m=[round(top[1] * px, 1), round(top[2] * px, 1)])


def nudge(rc, blob):
    """Slide a footprint up to 4 m so it best covers Google's visible roof."""
    H, W = blob.shape
    top = (-1, 0, 0)
    # Speed (2026-11-18): draw the footprint once on a canvas padded by the largest shift, then slide a
    # window over it. A whole pixel shift of the polygon is the same mask shifted, so the result is identical
    # to drawing it 441 times, about 100 times faster.
    PAD = 40
    big = draw.polygon2mask((H + 2 * PAD, W + 2 * PAD), rc + [PAD, PAD])
    for dr in range(-40, 41, 4):
        for dc in range(-40, 41, 4):
            m = big[PAD - dr:PAD - dr + H, PAD - dc:PAD - dc + W]
            score = (m & blob).sum() - 0.5 * (m & ~blob).sum()
            if score > top[0]:
                top = (score, dr, dc)
    return rc + [top[1], top[2]], (top[1], top[2])


def gmap_footprint(lat, lng, D, key):
    """TEST ONLY: the building shape Google draws in its plain map view (Maps Static API), in picture
    coordinates. Google's terms restrict creating content from Maps content, so this is for comparison,
    not for production, until that is cleared."""
    import io, urllib.request
    from skimage import measure as _m
    zoom, scale, size = 20, 2, 640
    q = [("center", "%f,%f" % (lat, lng)), ("zoom", str(zoom)), ("size", "%dx%d" % (size, size)), ("scale", str(scale)),
         ("maptype", "roadmap"),
         ("style", "feature:all|element:labels|visibility:off"),
         ("key", key)]
    data = NL.read("https://maps.googleapis.com/maps/api/staticmap?" + urllib.parse.urlencode(q))   # 0.1.2: time limit + retry
    im = np.array(Image.open(io.BytesIO(data)).convert("RGB")).astype(int)
    # buildings are drawn in one flat grey (233, 234, 239) on a lighter background
    red = (np.abs(im[..., 0] - 233) <= 4) & (np.abs(im[..., 1] - 234) <= 4) & (np.abs(im[..., 2] - 239) <= 4)
    red = ndi.binary_opening(red, iterations=2)
    if red.sum() < 200:
        return None, None, im
    lab, n = ndi.label(red)
    hh, ww = red.shape
    c = lab[hh // 2, ww // 2]
    if c == 0:
        _, idx = ndi.distance_transform_edt(lab == 0, return_indices=True)
        c = lab[idx[0][hh // 2, ww // 2], idx[1][hh // 2, ww // 2]]
    comp = ndi.binary_fill_holes(lab == c)
    # map picture pixel -> lat/lng (web mercator) -> Google Solar picture pixel
    world = 256 * (2 ** zoom) * scale
    cx = (lng + 180.0) / 360.0 * world
    siny = math.sin(math.radians(lat))
    cy = (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi)) * world
    mpp = 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom) / scale
    con = max(_m.find_contours(np.pad(comp.astype(float), 1), 0.5), key=len) - 1.0
    poly = _m.approximate_polygon(con, tolerance=0.25 / mpp)
    if np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]
    wx = cx + (poly[:, 1] - ww / 2.0)
    wy = cy + (poly[:, 0] - hh / 2.0)
    lngs = wx / world * 360.0 - 180.0
    lats = np.degrees(np.arctan(np.sinh(math.pi * (1 - 2 * wy / world))))
    tr = Transformer.from_crs(4326, D["epsg"], always_xy=True)
    xs, ys = tr.transform(lngs, lats)
    rc = np.c_[(D["y0"] - np.array(ys)) / D["px"], (np.array(xs) - D["x0"]) / D["px"]]
    per = sum(float(np.hypot(*(rc[(i + 1) % len(rc)] - rc[i]))) for i in range(len(rc))) * D["px"] * M2FT
    rc2, sh = nudge(rc, D["blob"])
    return rc2, dict(source="Google map view (test only)", corners=len(rc), perimeter_ft=round(per, 1),
                     area_sqft=round(float(comp.sum()) * mpp ** 2 * 10.7639),
                     shift_m=[round(sh[0] * D["px"], 1), round(sh[1] * D["px"], 1)]), im


def edge_height_labels(labels, zed, step_m):
    """Gable test from the roof edge height. Walk along an outline edge reading the height of the roof
    edge. Level = eave (gutter). Climbing to a peak and dropping = gable end (no gutter)."""
    z = np.array(zed, float)
    n = len(z)
    ok = np.isfinite(z)
    out = list(labels)
    if ok.sum() < 6:
        return out
    idx = np.arange(n)
    zi = np.interp(idx, idx[ok], z[ok])
    zi = ndi.median_filter(zi, size=5, mode="nearest")
    d = np.diff(zi)
    d[np.abs(d) > 0.6] = 0.0            # a sudden jump is a change of roof level, not a slope
    zc = np.r_[0.0, np.cumsum(d)]
    w = max(2, int(round(1.0 / step_m)))
    sl = np.zeros(n)
    for j in range(n):
        a_, b_ = max(0, j - w), min(n - 1, j + w)
        if b_ > a_:
            sl[j] = (zc[b_] - zc[a_]) / ((b_ - a_) * step_m)
    rake = np.abs(sl) >= RAKE_SLOPE
    gap = int(round(2.0 / step_m))      # the flat spot at the very peak of a gable
    j = 0
    while j < n:
        if not rake[j]:
            k2 = j
            while k2 < n and not rake[k2]:
                k2 += 1
            if j > 0 and k2 < n and (k2 - j) <= gap:
                rake[j:k2] = True
            j = k2
        else:
            j += 1
    # a real gable is wide and climbs a meaningful height. A chimney or vent at the eave is neither.
    j = 0
    while j < n:
        if rake[j]:
            k2 = j
            while k2 < n and rake[k2]:
                k2 += 1
            width_m = (k2 - j) * step_m
            seg = zc[j:k2]
            rise = float(seg.max() - min(seg[0], seg[-1]))
            span = float(seg.max() - seg.min())
            if width_m < GABLE_MIN_WIDTH_M or max(rise, span) < GABLE_MIN_RISE_M:
                rake[j:k2] = False
            j = k2
        else:
            j += 1
    for j in range(n):
        if not ok[j] or out[j] == "cover":
            continue
        out[j] = "gable" if rake[j] else "eave"
    return out


def analyze(D, poly_override=None):
    dsm, mask, px, blob = D["dsm"], D["mask"], D["px"], D["blob"]
    H, W = dsm.shape
    fin = np.isfinite(dsm)
    ring = ndi.binary_dilation(blob, iterations=60) & ~ndi.binary_dilation(blob, iterations=8) & fin & ~mask
    ground = float(np.percentile(dsm[ring], 15)) if ring.sum() > 100 else float(np.nanpercentile(dsm, 5))
    R = blob & fin & ((dsm - ground) > MIN_ROOF_HEIGHT_M)
    R = morphology.remove_small_objects(R, 200)

    maxf = ndi.maximum_filter(np.where(R, dsm, -np.inf), size=7)
    minf = ndi.minimum_filter(np.where(R, dsm, np.inf), size=7)
    step = R & ((maxf - minf) > STEP_M)
    V = R & ~step

    zs = E.norm_gauss(np.nan_to_num(dsm), V, 2.0)
    gr, gc = np.gradient(zs)
    gr, gc = gr / px, gc / px
    ok = ndi.binary_erosion(V, iterations=2) & np.isfinite(gr) & np.isfinite(gc)
    gr_s = E.norm_gauss(np.nan_to_num(gr), ok, 2.0)
    gc_s = E.norm_gauss(np.nan_to_num(gc), ok, 2.0)
    slope = np.hypot(gr_s, gc_s)
    pitch = np.degrees(np.arctan(slope))

    low = ok & (pitch < COVER_PITCH_DEG)
    cover = morphology.binary_opening(low, morphology.disk(6))
    cover = morphology.remove_small_objects(cover, int(MIN_COVER_M2 / px ** 2))
    cover = ndi.binary_dilation(cover, structure=morphology.disk(5)) & R
    steep = ok & (pitch >= COVER_PITCH_DEG) & ~cover

    # ---- main slope directions on this roof ----
    ang = np.degrees(np.arctan2(-gc_s, -gr_s)) % 360.0      # downhill direction, angle in (row, col) space
    hist, _ = np.histogram(ang[steep], bins=72, range=(0, 360))
    ker = np.array([1, 2, 3, 2, 1], float)
    hs = sum(np.roll(hist, k - 2) * ker[k] for k in range(5)) / ker.sum()
    order = np.argsort(hs)[::-1]
    peaks = []
    tot = max(hs.sum(), 1)
    for b in order:
        if hs[b] < 0.008 * tot:
            break
        a = (b + 0.5) * 5.0
        if all(abs((a - p + 180) % 360 - 180) >= PEAK_SEP_DEG for p in peaks):
            if hs[b] >= hs[(b - 1) % 72] and hs[b] >= hs[(b + 1) % 72]:
                peaks.append(a)
    # refine each peak to the weighted mean direction of its pixels
    cat = np.zeros((H, W), np.int32)        # 0 outside, -1 step, -2 cover, -3 other roof, >0 plane id
    cat[R] = -3
    cat[step] = -1
    cat[cover] = -2
    planes = []
    if peaks:
        pk = np.array(peaks)
        diff = np.abs((ang[..., None] - pk[None, None, :] + 180) % 360 - 180)
        near = np.argmin(diff, axis=2)
        nd = np.min(diff, axis=2)
        pid = 0
        for i in range(len(pk)):
            m = steep & (near == i) & (nd <= PEAK_TOL_DEG)
            m = morphology.binary_opening(m, morphology.disk(1))
            lab, n = ndi.label(m)
            for j in range(1, n + 1):
                pm = lab == j
                if pm.sum() * px ** 2 < MIN_PLANE_M2:
                    continue
                d = np.array([-np.mean(gr_s[pm]), -np.mean(gc_s[pm])])
                nrm = np.hypot(*d)
                if nrm < 1e-6:
                    continue
                d = d / nrm
                pid += 1
                cat[pm] = pid
                planes.append(dict(id=pid, d=d, t=np.array([d[1], -d[0]]), area_m2=float(pm.sum() * px ** 2),
                                   pitch=float(np.median(pitch[pm])), cent=[float(x) for x in ndi.center_of_mass(pm)]))

    # ---- structures: planes belong together when they touch along a real length (a shared ridge,
    # hip, valley, or an upper story wall). A corner touch or a patio cover bridge does not join them. ----
    parent = {p["id"]: p["id"] for p in planes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    passable = (cat == -1) | (cat == -3)      # step bands and unlabeled roof strips (ridges)
    dil = {}
    for p in planes:
        pm = cat == p["id"]
        rr_, cc_ = np.nonzero(pm)
        sl = (slice(max(rr_.min() - 14, 0), rr_.max() + 15), slice(max(cc_.min() - 14, 0), cc_.max() + 15))
        g = ndi.binary_dilation(pm[sl], iterations=10, mask=(pm | passable)[sl])
        full = np.zeros((H, W), bool)
        full[sl] = g
        dil[p["id"]] = full
    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            a_, b_ = planes[i]["id"], planes[j]["id"]
            con = dil[a_] & dil[b_]
            if con.sum() < 20:
                continue
            rr_, cc_ = np.nonzero(con)
            ext = math.hypot(rr_.max() - rr_.min(), cc_.max() - cc_.min()) * px
            if ext >= STRUCT_CONTACT_M:
                parent[find(a_)] = find(b_)
    for p in planes:
        p["struct"] = find(p["id"])
    slab = None

    # ---- low edge of every plane ----
    runs = []
    for p in planes:
        k, d, t = p["id"], p["d"], p["t"]
        rr, cc = np.nonzero(cat == k)
        r2 = np.clip(np.round(rr + 2 * d[0]).astype(int), 0, H - 1)
        c2 = np.clip(np.round(cc + 2 * d[1]).astype(int), 0, W - 1)
        sel = cat[r2, c2] != k
        rr, cc = rr[sel], cc[sel]
        if len(rr) == 0:
            continue
        n = len(rr)
        active = np.ones(n, bool)
        res = np.full(n, -99, np.int32)       # category found beyond the low edge
        hr = np.zeros(n, int)
        hc = np.zeros(n, int)
        for s in range(2, WALK_PX + 1):
            qr = np.round(rr + s * d[0]).astype(int)
            qc = np.round(cc + s * d[1]).astype(int)
            inb = (qr >= 0) & (qr < H) & (qc >= 0) & (qc < W)
            cv = np.zeros(n, np.int32)
            cv[inb] = cat[qr[inb], qc[inb]]
            back = active & (cv == k)
            active &= ~back
            hit = active & ((cv == 0) | (cv == -2) | ((cv > 0) & (cv != k)))
            res[hit] = cv[hit]
            hr[hit] = np.clip(qr[hit], 0, H - 1)
            hc[hit] = np.clip(qc[hit], 0, W - 1)
            active &= ~hit
        zb = zs[rr, cc]
        tag = np.full(n, "", dtype=object)
        # outside the roof: free edge, unless something higher (a tree) is sitting there
        out = res == 0
        if out.any():
            q2r = np.clip(np.round(hr + 4 * d[0]).astype(int), 0, H - 1)
            q2c = np.clip(np.round(hc + 4 * d[1]).astype(int), 0, W - 1)
            zo = np.fmax(np.nan_to_num(dsm[hr, hc], nan=-1e9), np.nan_to_num(dsm[q2r, q2c], nan=-1e9))
            tree = out & (zo > zb + TREE_RISE_M)
            tag[out & ~tree] = "free"
            tag[tree] = "tree"
        tag[res == -2] = "cover"
        oth = res > 0
        if oth.any():
            sid = {q["id"]: q["struct"] for q in planes}
            other_struct = np.array([sid.get(int(v), 0) for v in res])
            drop = zb - zs[hr, hc]
            diffs = oth & (other_struct != p["struct"]) & (drop > DROP_ONTO_LOWER_M)
            tag[diffs] = "faces"
            same = oth & ~diffs
            tag[same & (drop > DROP_ONTO_LOWER_M)] = "onto_lower"
            if os.environ.get("LDBG"):
                ml = same & (drop > DROP_ONTO_LOWER_M)
                if ml.sum() >= 3:
                    lp = pitch[hr[ml], hc[ml]]
                    ids, cnt = np.unique(res[ml], return_counts=True)
                    print("ONTO plane", k, "px", int(ml.sum()), "ft", round(ml.sum() * px * M2FT, 1), "drop m", round(float(np.median(drop[ml])), 1),
                          "landing pitch deg med", round(float(np.nanmedian(lp)), 1), "landing planes", dict(zip(ids.tolist(), cnt.tolist())),
                          "own pitch", round(p["pitch"], 1), "own area m2", round(p["area_m2"], 1), file=sys.stderr)
            tag[same & ~(drop > DROP_ONTO_LOWER_M)] = "valley"
        u = rr * t[0] + cc * t[1]
        v = rr * d[0] + cc * d[1]

        def extend(r_, c_, sgn):
            """Walk along the edge from a run end until the roof corner. Returns extra length in pixels."""
            got = 0
            pos = np.array([float(r_), float(c_)])
            for stp in range(1, int(EXTEND_MAX_M / px) + 1):
                q = pos + sgn * t
                found = None
                for o in (0, 1, -1, 2, -2, 3, -3):
                    qq = q + o * d
                    ri, ci = int(round(qq[0])), int(round(qq[1]))
                    if not (0 <= ri < H and 0 <= ci < W):
                        continue
                    cv = cat[ri, ci]
                    if cv == 0 or cv == -2 or (cv > 0 and cv != k):
                        continue
                    okk = False
                    for s2 in range(1, 13):
                        r3, c3 = int(round(qq[0] + s2 * d[0])), int(round(qq[1] + s2 * d[1]))
                        if not (0 <= r3 < H and 0 <= c3 < W):
                            break
                        c4 = cat[r3, c3]
                        if c4 == 0 or c4 == -2:
                            okk = True
                            break
                        if c4 > 0:
                            break
                    if okk:
                        found = qq
                        break
                if found is None:
                    break
                pos = found
                got = stp
            return got
        for kind, tags_ in (("eave", ("free", "tree", "cover", "faces")), ("onto_lower", ("onto_lower",))):
            m = np.isin(tag, tags_)
            if m.sum() < 3:
                continue
            uu, vv, tg, r_m, c_m = u[m], v[m], tag[m], rr[m], cc[m]
            o = np.argsort(uu)
            uu, vv, tg, r_m, c_m = uu[o], vv[o], tg[o], r_m[o], c_m[o]
            cuts = np.nonzero(np.diff(uu) > RUN_GAP_M / px)[0]
            starts = np.r_[0, cuts + 1]
            ends = np.r_[cuts, len(uu) - 1]
            for a, b in zip(starts, ends):
                ea = extend(r_m[a], c_m[a], -1.0) if kind == "eave" else 0
                eb = extend(r_m[b], c_m[b], 1.0) if kind == "eave" else 0
                Lm = (uu[b] - uu[a] + ea + eb) * px + END_ADD_M
                if Lm * M2FT < MIN_RUN_FT:
                    continue
                seg = tg[a:b + 1]
                vm = float(np.median(vv[a:b + 1]))
                p0 = (uu[a] - ea) * t + vm * d
                p1 = (uu[b] + eb) * t + vm * d
                runs.append(dict(plane=k, struct=p["struct"], kind=kind, length_ft=round(Lm * M2FT, 1),
                                 tree_share=round(float(np.mean(seg == "tree")), 2),
                                 over_cover_share=round(float(np.mean(seg == "cover")), 2),
                                 faces_other_share=round(float(np.mean(seg == "faces")), 2),
                                 p0=[float(p0[0]), float(p0[1])], p1=[float(p1[0]), float(p1[1])]))
    # ---- two planes can claim the same stretch of edge: trim the overlap from the shorter run ----
    ev = sorted([r for r in runs if r["kind"] == "eave"], key=lambda r: -r["length_ft"])
    kept = []
    for r in ev:
        a0, a1 = np.array(r["p0"]), np.array(r["p1"])
        L = np.hypot(*(a1 - a0))
        if L < 1:
            continue
        tt = (a1 - a0) / L
        lo, hi = 0.0, L
        for q in kept:
            b0, b1 = np.array(q["p0"]), np.array(q["p1"])
            Lq = np.hypot(*(b1 - b0))
            tq = (b1 - b0) / Lq
            if abs(float(np.dot(tt, tq))) < 0.95:
                continue
            mid = (a0 + a1) / 2
            w = mid - b0
            perp = abs(float(w[0] * tq[1] - w[1] * tq[0]))
            if perp > 10:        # more than 1 m apart: different edges
                continue
            s0, s1 = sorted([float(np.dot(b0 - a0, tt)), float(np.dot(b1 - a0, tt))])
            if s1 <= lo or s0 >= hi:
                continue
            if s0 <= lo and s1 >= hi:
                lo = hi
                break
            if s0 <= lo:
                lo = s1
            elif s1 >= hi:
                hi = s0
        if (hi - lo) * px * M2FT < MIN_RUN_FT:
            continue
        r = dict(r)
        r["p0"] = [float(x) for x in (a0 + lo * tt)]
        r["p1"] = [float(x) for x in (a0 + hi * tt)]
        r["length_ft"] = round(r["length_ft"] * (hi - lo) / L, 1)
        kept.append(r)
    runs = kept + [r for r in runs if r["kind"] != "eave"]
    # ================= outline pass =================
    # Walk Google's visible roof outline. For each short step, find the roof plane just inside the edge
    # and ask which way that plane drains. Toward the edge = eave (gutter). Along the edge = gable end.
    # Edges where the outline is following a TREE are thrown away. The gap is bridged with square
    # corners along the house's own wall directions, never by tracing the tree.
    pinfo = {p["id"]: p for p in planes}
    cons = measure.find_contours(np.pad(blob.astype(float), 1), 0.5)
    poly = measure.approximate_polygon(max(cons, key=len) - 1.0, tolerance=0.45 / px)
    if np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]
    if poly_override is not None:
        poly = np.array(poly_override, float)      # outline taken from an open building footprint instead of Google's mask
    N = len(poly)
    lens = [float(np.hypot(*(poly[(i + 1) % N] - poly[i]))) for i in range(N)]
    cen = poly.mean(axis=0)
    il = int(np.argmax(lens))
    tl = (poly[(il + 1) % N] - poly[il]) / lens[il]
    nl = np.array([tl[1], -tl[0]])
    sgn = 1.0 if float(np.dot((poly[il] + poly[(il + 1) % N]) / 2 - cen, nl)) > 0 else -1.0

    def at(arr, q, default=None):
        ri, ci = int(round(q[0])), int(round(q[1]))
        if 0 <= ri < H and 0 <= ci < W:
            return arr[ri, ci]
        return default

    # the house's wall directions, from the way its roof planes drain
    house_axis = None
    if planes:
        ang4 = sum(p["area_m2"] * np.exp(4j * math.atan2(p["d"][1], p["d"][0])) for p in planes)
        th = np.angle(ang4) / 4.0
        house_axis = (np.array([math.cos(th), math.sin(th)]), np.array([-math.sin(th), math.cos(th)]))

    # which outline edges are really a tree? A roof edge has lower ground outside. A tree edge has
    # something HIGHER than the roof just outside it.
    zs0 = E.norm_gauss(np.nan_to_num(dsm), blob & fin, 2.0)
    tree_share = []
    for i in range(N):
        p0, p1, L = poly[i], poly[(i + 1) % N], lens[i]
        if L < 1e-6:
            tree_share.append(0.0)
            continue
        t = (p1 - p0) / L
        n = np.array([t[1], -t[0]]) * sgn
        k = max(1, int(round(L * px / 0.2)))
        hits = tot = 0
        for j in range(k):
            p = p0 + t * L * (j + 0.5) / k
            zi = at(zs0, p - n * (0.6 / px))
            zo = [at(dsm, p + n * (dm / px)) for dm in (0.4, 0.8)]
            zo = [float(z) for z in zo if z is not None and np.isfinite(z)]
            if zi is None or not np.isfinite(zi) or not zo:
                continue
            tot += 1
            hits += int(max(zo) > float(zi) + TREE_RISE_M)
        tree_share.append(hits / tot if tot else 0.0)

    def off_axis(i):
        if house_axis is None or lens[i] < 1e-6:
            return False
        t = (poly[(i + 1) % N] - poly[i]) / lens[i]
        return max(abs(float(np.dot(t, house_axis[0]))), abs(float(np.dot(t, house_axis[1])))) < math.cos(math.radians(12))
    # A tree standing BESIDE a straight wall also reads as "higher outside". So only call it a tree edge
    # when the edge is also crooked: off the house's wall directions, or a short wiggle.
    occ = [tree_share[i] > 0.3 and (off_axis(i) or lens[i] * px < 2.5) for i in range(N)]
    for i in range(N):      # a short stub between two tree edges is tree too
        if not occ[i] and occ[i - 1] and occ[(i + 1) % N] and lens[i] * px < 1.5:
            occ[i] = True
    if all(occ) or poly_override is not None:
        occ = [False] * N              # a drawn footprint has no tree edges to throw away
    edt_out0 = ndi.distance_transform_edt(~blob)
    walk_n = 121 if poly_override is not None else 31

    per_edge = {}
    last_struct = 0
    jog_px_total = 0.0
    jog_count = 0
    for i in range(N):
        if occ[i]:
            continue
        p0, p1, L = poly[i], poly[(i + 1) % N], lens[i]
        if L < 1e-6:
            per_edge[i] = []
            continue
        t = (p1 - p0) / L
        n = np.array([t[1], -t[0]]) * sgn
        k = max(1, int(round(L * px / 0.2)))
        step_m = L * px / k
        raw, info, zed, insets = [], [], [], []
        for j in range(k):
            p = p0 + t * L * (j + 0.5) / k
            found, pid = None, 0
            # height of the roof edge here: step inward from a little outside the outline until Google's
            # roof mask starts, then read the roof height just past the blurred strip at the edge
            ze = np.nan
            for s_ in range(-12, 40):
                if at(R, p - n * s_, False):
                    vals = []
                    for dd in (3, 5, 7, 9):
                        q = p - n * (s_ + dd)
                        if at(R, q, False):
                            v = at(dsm, q)
                            if v is not None and np.isfinite(v):
                                vals.append(float(v))
                    if len(vals) >= 2:
                        ze = float(np.median(vals))
                    # if the mask only starts well inside the outline, or something taller sits right
                    # outside where it starts, a tree is covering the real edge: do not measure here
                    zo = [at(dsm, p - n * (s_ - dd)) for dd in (4, 8)]
                    zo = [float(v) for v in zo if v is not None and np.isfinite(v)]
                    if s_ > EDGE_MAX_INSET_PX or (zo and np.isfinite(ze) and max(zo) > ze + TREE_RISE_M):
                        ze = np.nan
                    break
            zed.append(ze)
            insets.append(float(s_) if np.isfinite(ze) else np.nan)
            unseen = poly_override is not None and not np.isfinite(ze)
            for s_ in range(2, 31 if unseen else walk_n):
                cv = at(cat, p - n * s_, 0)
                if cv > 0:
                    found, pid = "plane", int(cv)
                    break
                if cv == -2:
                    found = "cover"
                    break
            lbl, st = "unknown", last_struct
            if unseen and found is None:
                lbl = "eave"       # the photo cannot see this stretch. Most roof edges are eaves, so count it as gutter and flag it.
            elif found == "cover":
                lbl = "cover"
            elif found == "plane":
                pl = pinfo[pid]
                st = last_struct = pl["struct"]
                cval = float(np.dot(pl["d"], n))
                if cval >= EAVE_COS_O:
                    lbl = "eave"
                elif abs(cval) <= GABLE_COS_O:
                    lbl = "gable"
                elif cval < -GABLE_COS_O:
                    lbl = "high"
                else:
                    lbl = "unsure"
            raw.append(lbl)
            info.append((p, st, unseen))
        labels = E.smooth_labels(raw)
        if USE_EDGE_HEIGHT:
            labels = edge_height_labels(labels, zed, step_m)
        # jogs: where the real roof edge steps in or out from the drawn outline, the outline has smoothed
        # over a jog. Each jog adds its depth to the footage.
        ins = np.array(insets, float)
        w5 = 5
        jj = w5
        while jj <= len(ins) - w5:
            a_seg, b_seg = ins[jj - w5:jj], ins[jj:jj + w5]
            if np.isfinite(a_seg).all() and np.isfinite(b_seg).all():
                stp = abs(float(np.median(b_seg) - np.median(a_seg)))
                if stp >= JOG_MIN_PX and np.ptp(a_seg) <= 3 and np.ptp(b_seg) <= 3:
                    jog_px_total += stp
                    jog_count += 1
                    jj += w5
                    continue
            jj += 1
        per_edge[i] = [dict(r=float(p[0]), c=float(p[1]),
                            label=(lbl + "_tree") if (un and lbl in ("eave", "gable", "unsure")) else lbl,
                            len_m=step_m, struct=st)
                       for lbl, (p, st, un) in zip(labels, info)]

    def end_label(samples, tail):
        seg = samples[-8:] if tail else samples[:8]
        vals = [o["label"].replace("_tree", "") for o in seg if o["label"].replace("_tree", "") in ("eave", "gable")]
        if not vals:
            return "unsure"
        return max(set(vals), key=vals.count)

    def snap(t):
        if house_axis is None:
            return t
        best = max(house_axis, key=lambda e: abs(float(np.dot(t, e))))
        if abs(float(np.dot(t, best))) < math.cos(math.radians(20)):
            return t
        return best * (1.0 if float(np.dot(t, best)) > 0 else -1.0)

    outline = []
    tree_gaps = 0
    edt_in0 = ndi.distance_transform_edt(blob)
    vis = [i for i in range(N) if not occ[i]]
    for a_i, i in enumerate(vis):
        outline.extend(per_edge[i])
        j = vis[(a_i + 1) % len(vis)]
        if j == (i + 1) % N or len(vis) < 2:
            continue
        # tree gap between visible edge i and visible edge j: bridge it with square corners
        tree_gaps += 1
        a_pt, b_pt = poly[(i + 1) % N], poly[j]
        tA = snap((poly[(i + 1) % N] - poly[i]) / max(lens[i], 1e-6))
        e2 = np.array([-tA[1], tA[0]])
        delta = b_pt - a_pt
        du, dv = float(np.dot(delta, tA)), float(np.dot(delta, e2))
        st = per_edge[i][-1]["struct"] if per_edge[i] else last_struct
        la = end_label(per_edge[i], True) if per_edge[i] else "unsure"
        lb_ = end_label(per_edge[j], False) if per_edge[j] else "unsure"
        tB = (poly[(j + 1) % N] - poly[j]) / max(lens[j], 1e-6)
        if (abs(du) + abs(dv)) * px > MAX_TREE_GAP_M:
            # too big to bridge with any confidence: keep the stretch, marked unsure, so nothing silently vanishes
            kk = i
            while True:
                kk = (kk + 1) % N
                if kk == j:
                    break
                q0, q1 = poly[kk], poly[(kk + 1) % N]
                Ls = float(np.hypot(*(q1 - q0)))
                k = max(1, int(round(Ls * px / 0.2)))
                for jj in range(k):
                    q = q0 + (q1 - q0) * (jj + 0.5) / k
                    outline.append(dict(r=float(q[0]), c=float(q[1]), label="unsure_tree", len_m=Ls * px / k, struct=st))
            continue
        # two ways to square the corner. Take the one that stays out of the visible roof.
        cands = [a_pt + du * tA, a_pt + dv * e2]

        def inside_score(corner):
            tot_ = bad = 0
            for (q0, q1) in ((a_pt, corner), (corner, b_pt)):
                for f in np.linspace(0.05, 0.95, 12):
                    q = q0 + (q1 - q0) * f
                    tot_ += 1
                    bad += int(float(at(edt_in0, q, 0)) * px > 1.0)
            return bad / max(tot_, 1)
        corner = min(cands, key=inside_score)
        for (q0, q1) in ((a_pt, corner), (corner, b_pt)):
            Ls = float(np.hypot(*(q1 - q0)))
            if Ls * px < 0.05:
                continue
            ts = (q1 - q0) / Ls
            ns = np.array([ts[1], -ts[0]])
            mid = (q0 + q1) / 2
            if float(np.dot(cen - mid, ns)) > 0:      # make ns point away from the house
                ns = -ns
            fallback = la if abs(float(np.dot(ts, tA))) > 0.8 and np.hypot(*(q0 - a_pt)) < 1 else (
                lb_ if abs(float(np.dot(ts, tB))) > 0.8 and np.hypot(*(q1 - b_pt)) < 1 else "unsure")
            k = max(1, int(round(Ls * px / 0.2)))
            for jj in range(k):
                q = q0 + (q1 - q0) * (jj + 0.5) / k
                lab = fallback
                for s_ in range(1, 121):
                    cv = at(cat, q - ns * s_, 0)
                    if cv > 0:
                        cval = float(np.dot(pinfo[int(cv)]["d"], ns))
                        lab = "eave" if cval >= EAVE_COS_O else ("gable" if abs(cval) <= GABLE_COS_O else "unsure")
                        break
                outline.append(dict(r=float(q[0]), c=float(q[1]), label=lab + "_tree", len_m=Ls * px / k, struct=st))
    # a very short "gable" between gutter stretches is a corner artifact, not a gable end
    i = 0
    M = len(outline)
    while i < M:
        j = i
        while j < M and outline[j]["label"] == outline[i]["label"]:
            j += 1
        if outline[i]["label"] in ("gable", "unsure", "high") and sum(o["len_m"] for o in outline[i:j]) * M2FT < MIN_GABLE_FT:
            nb = [outline[i - 1]["label"] if i > 0 else outline[-1]["label"], outline[j % M]["label"]]
            pick = next((x_ for x_ in nb if x_ in ("eave", "eave_tree")), None)
            if pick:
                for o in outline[i:j]:
                    o["label"] = pick
        i = j
    # drop gutter stretches under 3 ft (a seen stretch and its continuation under a tree count as one)
    def is_g(o):
        return o["label"] in ("eave", "eave_tree")
    i = 0
    M = len(outline)
    while i < M:
        j = i
        while j < M and is_g(outline[j]) == is_g(outline[i]):
            j += 1
        if is_g(outline[i]) and sum(o["len_m"] for o in outline[i:j]) * M2FT < MIN_RUN_FT:
            for o in outline[i:j]:
                o["label"] = "short"
        i = j
    # ================= ridge following =================
    # A gable has a ridge that runs level all the way out to the roof edge. On a hip the ridge stops
    # short and the roof drops away. Ridges sit in the middle of the roof where the height data is cleanest.
    gables = []
    bigp = [p for p in planes if p["area_m2"] >= RIDGE_MIN_PLANE_M2]
    pm = {p["id"]: (cat == p["id"]) for p in bigp}
    pdil = {k: ndi.binary_dilation(v, iterations=RIDGE_GAP_PX) for k, v in pm.items()}
    for a_i in range(len(bigp)):
        for b_i in range(a_i + 1, len(bigp)):
            pa, pb = bigp[a_i], bigp[b_i]
            if float(np.dot(pa["d"], pb["d"])) > -0.3:
                continue
            con = pdil[pa["id"]] & pdil[pb["id"]] & R
            if os.environ.get("RDBG2"):
                print("PAIR", pa["id"], pb["id"], "areas", round(pa["area_m2"]), round(pb["area_m2"]), "pitch", round(pa["pitch"]), round(pb["pitch"]),
                      "dot", round(float(np.dot(pa["d"], pb["d"])), 2), "contact px", int(con.sum()), "cents", [round(x) for x in pa["cent"]], [round(x) for x in pb["cent"]], file=sys.stderr)
            if con.sum() < 40:
                continue
            rr_, cc_ = np.nonzero(con)
            pts_ = np.c_[rr_, cc_].astype(float)
            ctr = pts_.mean(axis=0)
            _, _, vt = np.linalg.svd(pts_ - ctr, full_matrices=False)
            ax = vt[0]
            if abs(float(np.dot(ax, pa["d"]))) > 0.5:
                continue
            ca, cb = np.array(pa["cent"]), np.array(pb["cent"])
            if float(np.dot(ca - ctr, pa["d"])) <= 0 or float(np.dot(cb - ctr, pb["d"])) <= 0:
                continue                       # planes drain toward the line: a valley, not a ridge
            tpar = (pts_ - ctr) @ ax
            if (tpar.max() - tpar.min()) * px < 1.0:
                continue
            zr = float(np.nanmedian(zs[rr_, cc_]))
            perp = np.array([-ax[1], ax[0]])
            for sg, tend in ((-1.0, float(tpar.min())), (1.0, float(tpar.max()))):
                E_ = ctr + ax * tend
                dv = ax * sg
                dist_px, beyond_tree = None, False
                for stp in range(2, int(RIDGE_SEARCH_M / px) + 1, 2):
                    q = E_ + dv * stp
                    if not at(R, q, False):
                        dist_px = stp
                        zq = [at(dsm, q + dv * e) for e in (2, 6, 10)]
                        zq = [float(v) for v in zq if v is not None and np.isfinite(v)]
                        beyond_tree = bool(zq and max(zq) > zr - 1.0)
                        break
                if os.environ.get("RDBG2"):
                    print("   END", pa["id"], pb["id"], "sg", sg, "E", [round(float(x)) for x in E_], "dist", None if dist_px is None else round(dist_px * px, 1), "ridge h", round(zr - ground, 1), file=sys.stderr)
                if os.environ.get("RDBG3"):
                    prof = []
                    for stp in range(0, 81, 5):
                        q = E_ + dv * stp
                        zz = at(dsm, q); inr = at(R, q, False); cid = at(cat, q, -1)
                        prof.append("%s%.1f/%s" % ("R" if inr else "x", (float(zz) - ground) if zz is not None and np.isfinite(zz) else -9, cid))
                    print("      PROF", " ".join(prof), file=sys.stderr)
                if dist_px is None:
                    continue                   # ridge end is far from any roof edge: a hip or a cross ridge
                if dist_px * px > RIDGE_EDGE_MAX_M:
                    continue                   # roof keeps going well past the ridge end: a hip
                # the deciding check: just inside the roof edge, is the roof still up at ridge height?
                # On a gable it is. On a hip it has dropped most of the way to the eave.
                zin = [at(dsm, E_ + dv * (dist_px - e)) for e in (7, 9, 11)]
                zin = [float(v) for v in zin if v is not None and np.isfinite(v)]
                if os.environ.get("RDBG"):
                    print("RIDGE", pa["id"], pb["id"], "end dist m", round(dist_px * px, 1), "ridge h", round(zr - ground, 1),
                          "h near edge", round(max(zin) - ground, 1) if zin else None, "tree", beyond_tree, file=sys.stderr)
                if not zin or max(zin) < zr - RIDGE_DROP_MAX_M:
                    continue
                widths = []
                for pl, side in ((pa, None), (pb, None)):
                    rr2, cc2 = np.nonzero(pm[pl["id"]])
                    q2 = np.c_[rr2, cc2].astype(float) - ctr
                    tk = q2 @ ax
                    near = np.abs(tk - tend) <= 25
                    pk = np.abs(q2 @ perp)
                    widths.append(float(np.percentile(pk[near], 98)) if near.sum() > 20 else 0.0)
                if min(widths) <= 0:
                    continue
                wft = (widths[0] + widths[1]) * px * M2FT
                if not (GABLE_MIN_FT <= wft <= GABLE_MAX_FT):
                    continue
                gables.append(dict(width_ft=round(wft, 1), hidden=beyond_tree, end=[float(E_[0] + dv[0] * dist_px), float(E_[1] + dv[1] * dist_px)],
                                   perp=[float(perp[0]), float(perp[1])],
                                   side_a_px=widths[0] if float(np.dot(ca - ctr, perp)) > 0 else widths[1],
                                   side_b_px=widths[1] if float(np.dot(ca - ctr, perp)) > 0 else widths[0],
                                   ridge_height_m=round(zr - ground, 1)))
    # the same gable can be found from two plane pairs: keep the widest of any that end at the same spot
    gables.sort(key=lambda g: -g["width_ft"])
    kept_g = []
    for g in gables:
        if all(np.hypot(g["end"][0] - h["end"][0], g["end"][1] - h["end"][1]) * px > 3.0 for h in kept_g):
            kept_g.append(g)
    gables = kept_g

    # interior runs: plane low edges that are NOT on the outline (they face another building or sit over a patio cover)
    edt_in = ndi.distance_transform_edt(blob)
    interior = []
    for r in runs:
        mid = (np.array(r["p0"]) + np.array(r["p1"])) / 2
        dist_m = float(at(edt_in, mid, 0)) * px
        if r["kind"] == "onto_lower" or dist_m > INTERIOR_MIN_M:
            interior.append(r)
    return dict(R=R, cover=cover, cat=cat, planes=planes, runs=runs, slab=slab, ground=ground,
                outline=outline, interior=interior, tree_repairs=tree_gaps, gables=gables,
                jog_ft=jog_px_total * px * M2FT, jog_count=jog_count, outline_corners=N,
                outline_perimeter_ft=sum(lens) * px * M2FT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--address")
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lng", type=float)
    ap.add_argument("--env", default=E.DEFAULT_ENV)
    ap.add_argument("--out", default=os.path.join(HERE, "results_v2"))
    ap.add_argument("--outline", default="google", choices=["google", "fema", "gmap"])
    a = ap.parse_args()
    key = E.read_key(a.env)
    os.makedirs(a.out, exist_ok=True)
    lat, lng, formatted = a.lat, a.lng, None
    if a.address:
        gj = E.get_json("https://maps.googleapis.com/maps/api/geocode/json?" +
                        urllib.parse.urlencode({"address": a.address, "key": key}))
        if gj.get("status") != "OK":
            sys.exit("Geocoding failed: %s" % gj.get("status"))
        loc = gj["results"][0]["geometry"]["location"]
        lat, lng = loc["lat"], loc["lng"]
        formatted = gj["results"][0].get("formatted_address")
    D = fetch(lat, lng, key)
    fp_info = None
    if a.outline == "fema":
        fp, fp_info = fema_footprint(lat, lng, D)
        if fp is None:
            sys.exit("No open footprint found for this address")
        A = analyze(D, poly_override=fp)
        a.name = a.name + "_fema"
    elif a.outline == "gmap":
        fp, fp_info, mapimg = gmap_footprint(lat, lng, D, key)
        if fp is None:
            sys.exit("Google map view shows no building shape here")
        A = analyze(D, poly_override=fp)
        a.name = a.name + "_gmap"
    else:
        A = analyze(D)
    px = D["px"]
    planes, runs = A["planes"], A["runs"]

    # structures, largest first; the one with the most plane area is the house
    area = {}
    for p in planes:
        area[p["struct"]] = area.get(p["struct"], 0.0) + p["area_m2"]
    order = sorted(area, key=lambda s: -area[s])
    names = {s: ("House" if i == 0 else "Structure %d" % (i + 1)) for i, s in enumerate(order)}
    items = []
    outl, interior = A["outline"], A["interior"]
    for s in order:
        o_e = sum(o["len_m"] for o in outl if o["struct"] == s and o["label"] == "eave") * M2FT
        o_t = sum(o["len_m"] for o in outl if o["struct"] == s and o["label"] == "eave_tree") * M2FT
        o_g = sum(o["len_m"] for o in outl if o["struct"] == s and o["label"] == "gable") * M2FT
        o_u = sum(o["len_m"] for o in outl if o["struct"] == s and o["label"] in ("unsure", "unknown")) * M2FT
        o_ut = sum(o["len_m"] for o in outl if o["struct"] == s and o["label"] == "unsure_tree") * M2FT
        o_gt = sum(o["len_m"] for o in outl if o["struct"] == s and o["label"] == "gable_tree") * M2FT
        iv = [r for r in interior if r["struct"] == s and r["kind"] == "eave"]
        ex = [r for r in interior if r["struct"] == s and r["kind"] == "onto_lower"]
        i_e = sum(r["length_ft"] for r in iv)
        items.append(dict(name=names[s], roof_area_sqft=round(area[s] * 10.7639),
                          gutter_ft=round(o_e + o_t + i_e, 1),
                          on_outline_ft=round(o_e, 1), hidden_by_tree_inferred_ft=round(o_t, 1),
                          interior_runs_ft=round(i_e, 1),
                          gable_ends_ft=round(o_g + o_gt, 1), unsure_ft=round(o_u, 1), unsure_under_tree_ft=round(o_ut, 1),
                          left_off_upper_onto_lower_ft=round(sum(r["length_ft"] for r in ex), 1)))
    res = dict(name=a.name, address=formatted or a.address, lat=lat, lng=lng,
               imagery_quality=D["quality"], imagery_date=D["date"],
               line_items=items, planes=len(planes), tree_gaps=A['tree_repairs'], outline_source=a.outline, footprint_info=fp_info,
               features=dict(
                   outline_perimeter_ft=round(A["outline_perimeter_ft"], 1),
                   outline_corners=A["outline_corners"],
                   jog_ft=round(A["jog_ft"], 1), jog_count=A["jog_count"],
                   gable_ridge_ft=round(sum(g["width_ft"] for g in A["gables"] if not g["hidden"]), 1),
                   gable_ridge_count=sum(1 for g in A["gables"] if not g["hidden"]),
                   gable_ridge_hidden_ft=round(sum(g["width_ft"] for g in A["gables"] if g["hidden"]), 1),
                   gable_edge_ft=round(sum(i["gable_ends_ft"] for i in items), 1),
                   unseen_ft=round(sum(i["hidden_by_tree_inferred_ft"] + i["unsure_under_tree_ft"] for i in items), 1),
                   interior_ft=round(sum(i["interior_runs_ft"] for i in items), 1),
                   imagery_quality=D["quality"], planes=len(planes),
                   plane_share=round(float((A["cat"] > 0).sum()) / max(float(A["R"].sum()), 1.0), 2),   # share of the roof that resolved into clean planes; low = height data is lumpy
                   roof_area_sqft=round(sum(i["roof_area_sqft"] for i in items)),
                   patio_cover_sqft=round(float(A["cover"].sum()) * px ** 2 * 10.7639)),
               gables=A["gables"],
               patio_cover_area_sqft=round(float(A["cover"].sum()) * px ** 2 * 10.7639),
               roof_touches_image_border=D["touches"])
    with open(os.path.join(a.out, a.name + ".json"), "w") as f:
        json.dump(dict(res, runs=runs), f, indent=2)

    # ---- picture ----
    S = 3
    rr, cc = np.nonzero(D["blob"])
    m = int(4 / px)
    H, W = D["dsm"].shape
    r0, r1, c0, c1 = max(0, rr.min() - m), min(H, rr.max() + m), max(0, cc.min() - m), min(W, cc.max() + m)
    base = D["rgb"][r0:r1, c0:c1].astype(float)
    cov = A["cover"][r0:r1, c0:c1]
    base[cov] = base[cov] * 0.55 + np.array([255, 255, 255]) * 0.45   # patio cover: washed out
    img = Image.fromarray(base.astype(np.uint8)).resize(((c1 - c0) * S, (r1 - r0) * S), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    palette = [(0, 220, 0), (0, 120, 255), (200, 0, 200), (0, 200, 200)]
    col = {s: palette[min(i, len(palette) - 1)] for i, s in enumerate(order)}

    def xy(p):
        return ((p[1] - c0) * S, (p[0] - r0) * S)
    hw = 3
    for o in outl:
        x, y = (o["c"] - c0) * S, (o["r"] - r0) * S
        lb = o["label"]
        if lb == "eave":
            d.ellipse([x - hw, y - hw, x + hw, y + hw], fill=col.get(o["struct"], palette[0]))
        elif lb == "eave_tree":
            d.ellipse([x - hw - 2, y - hw - 2, x + hw + 2, y + hw + 2], fill=(255, 255, 255))
            d.ellipse([x - hw, y - hw, x + hw, y + hw], fill=col.get(o["struct"], palette[0]))
        elif lb == "gable":
            d.ellipse([x - hw, y - hw, x + hw, y + hw], fill=(255, 0, 0))
        elif lb == "gable_tree":
            d.ellipse([x - hw - 2, y - hw - 2, x + hw + 2, y + hw + 2], fill=(255, 255, 255))
            d.ellipse([x - hw, y - hw, x + hw, y + hw], fill=(255, 0, 0))
        elif lb == "unsure_tree":
            d.ellipse([x - hw - 2, y - hw - 2, x + hw + 2, y + hw + 2], fill=(255, 255, 255))
            d.ellipse([x - hw, y - hw, x + hw, y + hw], fill=(255, 255, 0))
        elif lb in ("unsure", "unknown"):
            d.ellipse([x - hw, y - hw, x + hw, y + hw], fill=(255, 255, 0))
    for g in A["gables"]:
        e_ = np.array(g["end"]); pp = np.array(g["perp"])
        g0 = xy(e_ + pp * g["side_a_px"]); g1 = xy(e_ - pp * g["side_b_px"])
        d.line([g0, g1], fill=(255, 255, 255), width=9)
        d.line([g0, g1], fill=(255, 0, 255) if not g["hidden"] else (150, 0, 150), width=5)
    for r in interior:
        a0, a1 = xy(r["p0"]), xy(r["p1"])
        if r["kind"] == "onto_lower":
            d.line([a0, a1], fill=(255, 140, 0), width=3)
        else:
            d.line([a0, a1], fill=col.get(r["struct"], palette[0]), width=6)
    lines = ["%s" % (formatted or a.address or a.name)]
    for it in items:
        lines.append("%s: %.0f ft gutter (%.0f seen, %.0f inferred under trees, %.0f interior)  gable %.0f ft  unsure %.0f ft seen + %.0f ft under trees" % (
            it["name"], it["gutter_ft"], it["on_outline_ft"], it["hidden_by_tree_inferred_ft"], it["interior_runs_ft"],
            it["gable_ends_ft"], it["unsure_ft"], it["unsure_under_tree_ft"]))
    fe = res["features"]
    lines.append("Ridge test: %d gables, %.0f ft (+%.0f ft possible under trees)   Edge height test: %.0f ft   Jogs: %d adding %.0f ft" % (
        fe["gable_ridge_count"], fe["gable_ridge_ft"], fe["gable_ridge_hidden_ft"], fe["gable_edge_ft"], fe["jog_count"], fe["jog_ft"]))
    lines.append("green = gutter   RED dots = gable by edge height   PINK bar = gable by ridge test   yellow = unsure   blue = separate structure")
    lines.append("white border = under a tree, inferred   orange = left off (upper roof onto lower roof)   washed out = patio cover, left off")
    bh = 16 * len(lines) + 8
    out = Image.new("RGB", (max(img.width, 760), img.height + bh), (255, 255, 255))
    out.paste(img, (0, bh))
    d2 = ImageDraw.Draw(out)
    for i, ln in enumerate(lines):
        d2.text((6, 4 + 16 * i), ln, fill=(0, 0, 0))
    out.save(os.path.join(a.out, a.name + ".png"))
    if os.environ.get("EDBG"):
        cat = A["cat"][r0:r1, c0:c1]
        rng = np.random.RandomState(3)
        lut = rng.randint(60, 255, size=(max(cat.max(), 1) + 1, 3))
        dbg = np.zeros(cat.shape + (3,), np.uint8)
        dbg[cat > 0] = lut[cat[cat > 0]]
        dbg[cat == -1] = (0, 0, 0)
        dbg[cat == -2] = (255, 255, 255)
        dbg[cat == -3] = (90, 90, 90)
        dbg[cat == 0] = (30, 0, 60)
        Image.fromarray(dbg).resize(((c1 - c0) * 2, (r1 - r0) * 2), Image.NEAREST).save(os.path.join(a.out, a.name + "_debug.png"))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
