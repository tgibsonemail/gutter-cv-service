#!/usr/bin/env python3
"""
Gutter eave footage test.
Idea: an outline edge gets a gutter only if the roof slopes DOWN toward it (eave).
If the roof slopes ALONG the edge, it is a gable (rake) edge and gets no gutter.
Data: Google Solar API height map (DSM), roof mask and aerial photo at 0.1 m per pixel.

Usage:
  python3 eave_test.py --name house1 --address "123 Main St, Dallas, TX"
  python3 eave_test.py --name house1 --lat 32.84 --lng -96.78
The API key is read from the env file (GOOGLE_MAPS_API_KEY=...), never stored in this code.
"""
import argparse, json, math, os, sys, io, urllib.request, urllib.parse
import numpy as np
import logging
logging.getLogger('tifffile').setLevel(logging.ERROR)
import tifffile
from scipy import ndimage as ndi
from skimage import measure, morphology
from PIL import Image, ImageDraw
from pyproj import Transformer

M2FT = 3.28084
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ENV = os.path.join(HERE, "..", "..", "context", "EG Fable Test", "google.env")

# ---- thresholds (tunable) ----
FLAT_PITCH_DEG = 4.0     # below this the roof is treated as flat
EAVE_COS = 0.60          # downhill direction within about 53 deg of the outward normal = eave
RAKE_COS = 0.40          # downhill direction within about 66..114 deg of the normal = gable edge
SIMPLIFY_M = 0.45        # outline straightening tolerance
STEP_M = 0.2             # sample spacing along each edge
STEP_DROP_M = 1.0        # interior height drop that counts as an upper story edge
TREE_RISE_M = 0.6        # if the surface just outside the outline is this much higher than the roof, a tree is in the way
TREE_EDGE_SHARE = 0.3    # share of an edge that must fail the tree check before it is treated as tree
MAX_EXTEND_M = 12.0      # longest hidden stretch the repair will bridge


def read_key(env_path):
    with open(env_path) as f:
        for line in f:
            if line.strip().startswith("GOOGLE_MAPS_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("No GOOGLE_MAPS_API_KEY in " + env_path)


def get_json(url):
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf8", "ignore")
        try:
            msg = json.loads(body)["error"]["message"]
        except Exception:
            msg = body[:300]
        sys.exit("Google API error %s: %s" % (e.code, msg))


def get_tif(url, key):
    with urllib.request.urlopen(url + "&key=" + key, timeout=120) as r:
        data = r.read()
    tf = tifffile.TiffFile(io.BytesIO(data))
    page = tf.pages[0]
    arr = page.asarray()
    tags = {t.name: t.value for t in page.tags.values()}
    return arr, tags


def epsg_from_tags(tags):
    gk = tags.get("GeoKeyDirectoryTag")
    if gk is None:
        return None
    gk = list(gk)
    for i in range(4, len(gk), 4):
        if gk[i] == 3072:  # ProjectedCSTypeGeoKey
            return int(gk[i + 3])
    return None


def norm_gauss(a, m, sigma):
    """Gaussian smoothing that only uses pixels where m is True."""
    w = ndi.gaussian_filter(m.astype(float), sigma)
    v = ndi.gaussian_filter(np.where(m, a, 0.0), sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = v / w
    out[w < 1e-3] = np.nan
    return out


def classify(g, n):
    """g = uphill gradient (per meter) in (row, col); n = unit direction pointing off the roof."""
    s = float(np.hypot(g[0], g[1]))
    if not np.isfinite(s):
        return "unknown", 0.0, 0.0
    pitch = math.degrees(math.atan(s))
    if pitch < FLAT_PITCH_DEG:
        return "flat", pitch, 0.0
    d = -g / s  # downhill unit vector
    c = float(d[0] * n[0] + d[1] * n[1])
    if c >= EAVE_COS:
        return "eave", pitch, c
    if abs(c) <= RAKE_COS:
        return "rake", pitch, c
    if c < -RAKE_COS:
        return "high", pitch, c   # roof slopes away from this edge (top of a shed roof)
    return "unsure", pitch, c


def smooth_labels(labels, win=5):
    """Mode filter along the edge, then absorb short flat or unknown runs into neighbors."""
    n = len(labels)
    if n == 0:
        return labels
    out = list(labels)
    h = win // 2
    for i in range(n):
        seg = labels[max(0, i - h):min(n, i + h + 1)]
        vals, cnt = np.unique(seg, return_counts=True)
        out[i] = str(vals[np.argmax(cnt)])
    # short runs of flat/unknown/unsure (under 1.5 m) take the neighbor label
    i = 0
    maxrun = int(round(1.5 / STEP_M))
    while i < n:
        j = i
        while j < n and out[j] == out[i]:
            j += 1
        if out[i] in ("flat", "unknown", "unsure") and (j - i) <= maxrun:
            left = out[i - 1] if i > 0 else None
            right = out[j] if j < n else None
            repl = left or right
            if left and right and left != right:
                repl = left if left in ("eave", "rake") else right
            if repl:
                for k in range(i, j):
                    out[k] = repl
        i = j
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--address")
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lng", type=float)
    ap.add_argument("--env", default=DEFAULT_ENV)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    key = read_key(a.env)
    os.makedirs(a.out, exist_ok=True)

    lat, lng, formatted = a.lat, a.lng, None
    if a.address:
        gj = get_json("https://maps.googleapis.com/maps/api/geocode/json?" +
                      urllib.parse.urlencode({"address": a.address, "key": key}))
        if gj.get("status") != "OK":
            sys.exit("Geocoding failed: %s %s" % (gj.get("status"), gj.get("error_message", "")))
        loc = gj["results"][0]["geometry"]["location"]
        lat, lng = loc["lat"], loc["lng"]
        formatted = gj["results"][0].get("formatted_address")

    bi = get_json("https://solar.googleapis.com/v1/buildingInsights:findClosest?" +
                  urllib.parse.urlencode({"location.latitude": lat, "location.longitude": lng,
                                          "requiredQuality": "MEDIUM", "key": key}))
    c = bi["center"]
    bb = bi["boundingBox"]
    clat, clng = c["latitude"], c["longitude"]
    dlat = (bb["ne"]["latitude"] - bb["sw"]["latitude"]) * 111320.0
    dlng = (bb["ne"]["longitude"] - bb["sw"]["longitude"]) * 111320.0 * math.cos(math.radians(clat))
    radius = min(max(0.5 * math.hypot(dlat, dlng) + 8.0, 20.0), 75.0)
    segs = bi.get("solarPotential", {}).get("roofSegmentStats", [])

    for attempt in range(3):
        dl = get_json("https://solar.googleapis.com/v1/dataLayers:get?" +
                      urllib.parse.urlencode({"location.latitude": clat, "location.longitude": clng,
                                              "radiusMeters": round(radius, 1), "view": "IMAGERY_LAYERS",
                                              "requiredQuality": "MEDIUM", "pixelSizeMeters": 0.1,
                                              "key": key}))
        dsm, tags = get_tif(dl["dsmUrl"], key)
        mask, _ = get_tif(dl["maskUrl"], key)
        rgb, _ = get_tif(dl["rgbUrl"], key)
        if rgb.ndim == 3 and rgb.shape[0] in (3, 4) and rgb.shape[2] not in (3, 4):
            rgb = np.moveaxis(rgb, 0, -1)
        rgb = rgb[..., :3].astype(np.uint8)
        dsm = dsm.astype(float)
        dsm[dsm < -1000] = np.nan
        mask = mask.astype(bool)

        if "ModelTransformationTag" in tags:
            mt = tags["ModelTransformationTag"]
            px, x0, y0 = float(mt[0]), float(mt[3]), float(mt[7])
        else:
            px = float(tags["ModelPixelScaleTag"][0])
            tie = tags["ModelTiepointTag"]
            x0, y0 = float(tie[3]), float(tie[4])
        epsg = epsg_from_tags(tags)
        H, W = dsm.shape
        if epsg:
            tr = Transformer.from_crs(4326, epsg, always_xy=True)

            def ll2rc(la, lo):
                x, y = tr.transform(lo, la)
                return (y0 - y) / px, (x - x0) / px
        else:
            def ll2rc(la, lo):
                return (H / 2 - (la - clat) * 111320.0 / px,
                        W / 2 + (lo - clng) * 111320.0 * math.cos(math.radians(clat)) / px)

        # ---- pick the building at the center ----
        lab, nlab = ndi.label(mask)
        cr, cc = ll2rc(clat, clng)
        cr_i, cc_i = int(round(min(max(cr, 0), H - 1))), int(round(min(max(cc, 0), W - 1)))
        if nlab == 0:
            sys.exit("No roof found in the mask")
        if lab[cr_i, cc_i] > 0:
            target = lab[cr_i, cc_i]
        else:
            dist, idx = ndi.distance_transform_edt(lab == 0, return_indices=True)
            target = lab[idx[0][cr_i, cc_i], idx[1][cr_i, cc_i]]
        comp = lab == target
        comp = morphology.binary_opening(comp, morphology.disk(2))
        lab2, n2 = ndi.label(comp)
        if n2 > 1:
            sizes = ndi.sum(comp, lab2, range(1, n2 + 1))
            comp = lab2 == (1 + int(np.argmax(sizes)))
        filled = ndi.binary_fill_holes(comp)
        rr_, cc__ = np.nonzero(filled)
        touches_border = bool(rr_.min() <= 4 or cc__.min() <= 4 or rr_.max() >= H - 5 or cc__.max() >= W - 5)
        if not touches_border or radius >= 99:
            break
        radius = min(radius * 1.6, 100.0)   # the roof ran off the picture: fetch a wider area

    # ---- roof slope field ----
    valid = filled & np.isfinite(dsm)
    zs = norm_gauss(np.nan_to_num(dsm), valid, 2.0)
    inner = ndi.binary_erosion(valid, iterations=4)
    gr, gc = np.gradient(zs)
    gr, gc = gr / px, gc / px
    ok = inner & np.isfinite(gr) & np.isfinite(gc)
    gr_s = norm_gauss(np.nan_to_num(gr), ok, 3.0)
    gc_s = norm_gauss(np.nan_to_num(gc), ok, 3.0)
    gr_s[~ok] = np.nan
    gc_s[~ok] = np.nan

    def grad_at(r, c_):
        ri, ci = int(round(r)), int(round(c_))
        if 0 <= ri < H and 0 <= ci < W and ok[ri, ci]:
            return np.array([gr_s[ri, ci], gc_s[ri, ci]])
        return None

    # ---- Google roof segments, used as a second opinion ----
    seg_px = []
    for s in segs:
        b = s["boundingBox"]
        r1, c1 = ll2rc(b["ne"]["latitude"], b["sw"]["longitude"])
        r2, c2 = ll2rc(b["sw"]["latitude"], b["ne"]["longitude"])
        seg_px.append((min(r1, r2), max(r1, r2), min(c1, c2), max(c1, c2),
                       s.get("azimuthDegrees", 0.0), s.get("pitchDegrees", 0.0)))

    def google_says(r, c_, n):
        best = None
        for (ra, rb, ca, cb, az, pit) in seg_px:
            if ra <= r <= rb and ca <= c_ <= cb:
                area = (rb - ra) * (cb - ca)
                if best is None or area < best[0]:
                    best = (area, az, pit)
        if best is None:
            return None
        if best[2] < FLAT_PITCH_DEG:
            return "flat"
        bearing = math.degrees(math.atan2(n[1], -n[0])) % 360.0
        diff = abs((best[1] - bearing + 180) % 360 - 180)
        if diff <= 50:
            return "eave"
        if 55 <= diff <= 125:
            return "rake"
        return "other"

    # ---- outline, straightened ----
    pad = np.pad(filled.astype(float), 1)
    cons = measure.find_contours(pad, 0.5)
    con = max(cons, key=len) - 1.0
    poly = measure.approximate_polygon(con, tolerance=SIMPLIFY_M / px)
    if np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]

    # ---- which way is "off the roof" (from the polygon's winding direction) ----
    N0 = len(poly)
    lens = [float(np.hypot(*(poly[(i + 1) % N0] - poly[i]))) for i in range(N0)]
    il = int(np.argmax(lens))
    tl = (poly[(il + 1) % N0] - poly[il]) / lens[il]
    nl = np.array([tl[1], -tl[0]])
    tp = (poly[il] + poly[(il + 1) % N0]) / 2 + 4 * nl
    ri, ci = int(round(tp[0])), int(round(tp[1]))
    sgn = -1.0 if (0 <= ri < H and 0 <= ci < W and filled[ri, ci]) else 1.0

    def out_normal(t):
        return np.array([t[1], -t[0]]) * sgn

    def z_at(arr, q):
        ri, ci = int(round(q[0])), int(round(q[1]))
        if 0 <= ri < H and 0 <= ci < W and np.isfinite(arr[ri, ci]):
            return float(arr[ri, ci])
        return None

    # ---- tree check: a real roof edge has lower ground outside it. If what is just
    # outside the outline is HIGHER than the roof, a tree is covering the roof there and
    # the outline is following the tree, not the roofline. ----
    occ = []
    for i in range(N0):
        p0, p1 = poly[i], poly[(i + 1) % N0]
        L = lens[i]
        if L < 1e-6:
            occ.append(False)
            continue
        t = (p1 - p0) / L
        n = out_normal(t)
        k = max(1, int(round(L * px / STEP_M)))
        hits = tot = 0
        for j in range(k):
            p = p0 + t * L * (j + 0.5) / k
            zi = z_at(zs, p - n * (0.6 / px))
            zo = [z_at(dsm, p + n * (dm / px)) for dm in (0.5, 0.9, 1.3)]
            zo = [z for z in zo if z is not None]
            if zi is None or not zo:
                continue
            tot += 1
            hits += int(max(zo) > zi + TREE_RISE_M)
        occ.append(tot > 0 and hits / tot > TREE_EDGE_SHARE)
        if os.environ.get('EDBG'): print('EDGE', i, 'len_ft', round(L*px*M2FT,1), 'from', [int(x) for x in p0], 'to', [int(x) for x in p1], 'occ', hits, '/', tot, file=sys.stderr)

    # ---- repair: drop tree edges and extend the visible roof edges on each side until they meet ----
    repaired_runs = 0
    if any(occ) and not all(occ):
        vis = [i for i in range(N0) if not occ[i]]
        st = {i: poly[i].copy() for i in vis}
        en = {i: poly[(i + 1) % N0].copy() for i in vis}
        for a_i, i in enumerate(vis):
            j = vis[(a_i + 1) % len(vis)]
            if j == (i + 1) % N0 or i == j:
                continue
            A0, A1, B0, B1 = st[i], en[i], st[j], en[j]
            da, db = A1 - A0, B1 - B0
            den = da[0] * db[1] - da[1] * db[0]
            cosang = abs(float(np.dot(da, db))) / (np.hypot(*da) * np.hypot(*db) + 1e-9)
            if abs(den) < 1e-9 or cosang > math.cos(math.radians(20)):
                repaired_runs += 1      # same wall line on both sides of the tree: join straight across
                continue
            w = B0 - A0
            ta = (w[0] * db[1] - w[1] * db[0]) / den
            tb = (w[0] * da[1] - w[1] * da[0]) / den
            X = A0 + ta * da
            lim = MAX_EXTEND_M / px
            if ta > 0.5 and tb < 0.5 and np.hypot(*(X - A1)) < lim and np.hypot(*(X - B0)) < lim:
                en[i] = X
                st[j] = X
            repaired_runs += 1
        pts_new = []
        for a_i, i in enumerate(vis):
            for q in (st[i], en[i]):
                if not pts_new or np.hypot(*(q - pts_new[-1])) > 1.0:
                    pts_new.append(q)
        if len(pts_new) > 2 and np.hypot(*(pts_new[0] - pts_new[-1])) <= 1.0:
            pts_new = pts_new[:-1]
        if len(pts_new) >= 3:
            poly = np.array(pts_new)
    edt_out = ndi.distance_transform_edt(~filled)

    samples = []   # (r, c, label, pitch, edge index, hidden by tree)
    edges = []
    totals = {}
    hidden = {}
    agree = 0
    compared = 0
    for i in range(len(poly)):
        p0, p1 = poly[i], poly[(i + 1) % len(poly)]
        v = p1 - p0
        L = float(np.hypot(v[0], v[1]))
        if L < 1e-6:
            continue
        t = v / L
        n = out_normal(t)
        k = max(1, int(round(L * px / STEP_M)))
        raw, pts, pitches = [], [], []
        for j in range(k):
            p = p0 + t * L * (j + 0.5) / k
            gs = []
            for dist_m in (0.7, 0.9, 1.1, 1.3, 1.6, 2.0):
                q = p - n * (dist_m / px)
                g = grad_at(q[0], q[1])
                if g is not None:
                    gs.append(g)
                if len(gs) >= 3:
                    break
            if gs:
                g = np.mean(gs, axis=0)
                lbl, pitch, cosv = classify(g, n)
            else:
                lbl, pitch = "unknown", 0.0
            raw.append(lbl)
            pts.append(p)
            pitches.append(pitch)
        labels = smooth_labels(raw)
        # parts hidden under a tree have no readable slope: carry the label over from the visible part
        known = [x for x in labels if x != "unknown"]
        if not known:
            prev = samples[-1][2] if samples else "unsure"
            labels = [prev if prev != "unknown" else "unsure"] * len(labels)
        else:
            last = None
            for q_i in range(len(labels)):
                if labels[q_i] == "unknown":
                    labels[q_i] = last
                else:
                    last = labels[q_i]
            nxt = None
            for q_i in range(len(labels) - 1, -1, -1):
                if labels[q_i] is None:
                    labels[q_i] = nxt
                else:
                    nxt = labels[q_i]
        seg_len_m = L * px / k
        for p, lbl, pitch in zip(pts, labels, pitches):
            ri, ci = int(round(min(max(p[0], 0), H - 1))), int(round(min(max(p[1], 0), W - 1)))
            hid = bool(edt_out[ri, ci] > 0.5 / px)
            samples.append((float(p[0]), float(p[1]), lbl, pitch, i, hid))
            totals[lbl] = totals.get(lbl, 0.0) + seg_len_m
            if hid:
                hidden[lbl] = hidden.get(lbl, 0.0) + seg_len_m
            if lbl in ("eave", "rake"):
                q = p - n * (1.2 / px)
                gsay = google_says(q[0], q[1], n)
                if gsay in ("eave", "rake"):
                    compared += 1
                    agree += int(gsay == lbl)
        edges.append({"from": [float(p0[0]), float(p0[1])], "to": [float(p1[0]), float(p1[1])],
                      "length_ft": round(L * px * M2FT, 1)})

    # ---- interior eaves: an upper roof that drains onto a lower roof ----
    zraw = np.where(valid, dsm, np.inf)
    zmin = ndi.minimum_filter(zraw, size=13)
    drop = np.where(valid, dsm - zmin, 0.0)
    edt = ndi.distance_transform_edt(filled)
    band = (drop > STEP_DROP_M) & (edt > 1.5 / px)
    band = morphology.remove_small_objects(band, 30)
    skel = morphology.skeletonize(band)
    zbig = norm_gauss(np.nan_to_num(dsm), valid, 5.0)
    br, bc = np.gradient(zbig)
    interior = []
    slab, sn = ndi.label(skel, structure=np.ones((3, 3)))
    for li in range(1, sn + 1):
        rr, cc_ = np.nonzero(slab == li)
        if len(rr) * px < 2.0:
            continue
        for r, c_ in zip(rr, cc_):
            up = np.array([br[r, c_], bc[r, c_]])
            m = np.hypot(up[0], up[1])
            if not np.isfinite(m) or m < 1e-6:
                continue
            up = up / m
            gs = []
            for dist_m in (1.0, 1.3, 1.6, 2.0):
                q = np.array([r, c_]) + up * (dist_m / px)
                g = grad_at(q[0], q[1])
                if g is not None:
                    gs.append(g)
            if not gs:
                continue
            lbl, pitch, cosv = classify(np.mean(gs, axis=0), -up)
            if lbl == "eave":
                interior.append((int(r), int(c_)))
    interior_m = len(interior) * px * 1.1

    # ---- numbers ----
    eave_ft = totals.get("eave", 0.0) * M2FT
    rake_ft = totals.get("rake", 0.0) * M2FT
    unsure_ft = (totals.get("unsure", 0.0) + totals.get("unknown", 0.0)) * M2FT
    flat_ft = totals.get("flat", 0.0) * M2FT
    high_ft = totals.get("high", 0.0) * M2FT
    perim_ft = sum(totals.values()) * M2FT
    interior_ft = interior_m * M2FT
    pit = [s[3] for s in samples if s[2] in ("eave", "rake")]
    res = {
        "name": a.name, "address": formatted or a.address, "lat": lat, "lng": lng,
        "imagery_quality": dl.get("imageryQuality"), "imagery_date": dl.get("imageryDate"),
        "pixel_m": px, "outline_perimeter_ft": round(perim_ft, 1),
        "eave_ft_outline": round(eave_ft, 1), "eave_ft_interior_upper_story_approx": round(interior_ft, 1),
        "eave_ft_total": round(eave_ft + interior_ft, 1),
        "eave_ft_hidden_by_tree_inferred": round(hidden.get("eave", 0.0) * M2FT, 1),
        "tree_repairs": repaired_runs,
        "gable_rake_ft": round(rake_ft, 1), "flat_edge_ft": round(flat_ft, 1),
        "high_edge_ft": round(high_ft, 1), "unsure_ft": round(unsure_ft, 1),
        "median_pitch_deg": round(float(np.median(pit)), 1) if pit else None,
        "google_segments": len(segs),
        "google_agreement": round(agree / compared, 2) if compared else None,
        "google_compared_samples": compared,
        "roof_touches_image_border": touches_border,
        "outline_corners": len(poly),
    }
    with open(os.path.join(a.out, a.name + ".json"), "w") as f:
        json.dump(res, f, indent=2)

    # ---- picture ----
    S = 3
    r0, r1 = np.nonzero(filled)[0].min(), np.nonzero(filled)[0].max()
    c0, c1 = np.nonzero(filled)[1].min(), np.nonzero(filled)[1].max()
    r0, r1 = min(r0, int(poly[:, 0].min())), max(r1, int(poly[:, 0].max()))
    c0, c1 = min(c0, int(poly[:, 1].min())), max(c1, int(poly[:, 1].max()))
    m = int(4 / px)
    r0, r1, c0, c1 = max(0, r0 - m), min(H, r1 + m), max(0, c0 - m), min(W, c1 + m)
    img = Image.fromarray(rgb[r0:r1, c0:c1]).resize(((c1 - c0) * S, (r1 - r0) * S), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    col = {"eave": (0, 220, 0), "rake": (255, 0, 0), "unsure": (255, 255, 0), "unknown": (255, 255, 0),
           "flat": (0, 120, 255), "high": (255, 140, 0)}
    hw = (STEP_M / px) * S / 2
    for (r, c_, lbl, pitch, ei, hid) in samples:
        x, y = (c_ - c0) * S, (r - r0) * S
        colr = col.get(lbl, (255, 255, 255))
        if hid:
            d.ellipse([x - hw - 3, y - hw - 3, x + hw + 3, y + hw + 3], fill=(255, 255, 255))
        d.ellipse([x - hw - 1, y - hw - 1, x + hw + 1, y + hw + 1], fill=colr)
    for (r, c_) in interior:
        x, y = (c_ - c0) * S, (r - r0) * S
        d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(0, 255, 255))
    lines = ["%s" % (formatted or a.address or a.name),
             "EAVE (gutter) %.0f ft outline + %.0f ft upper story = %.0f ft" % (eave_ft, interior_ft, eave_ft + interior_ft),
             "GABLE (no gutter) %.0f ft   unsure %.0f ft   flat %.0f ft" % (rake_ft, unsure_ft, flat_ft),
             "green=eave  red=gable  cyan=upper story eave  yellow=unsure  blue=flat",
             "white border = hidden under a tree, inferred (%.0f ft of the eave total)" % (hidden.get("eave", 0.0) * M2FT)]
    bh = 16 * len(lines) + 8
    out = Image.new("RGB", (max(img.width, 560), img.height + bh), (255, 255, 255))
    out.paste(img, (0, bh))
    d2 = ImageDraw.Draw(out)
    for i, ln in enumerate(lines):
        d2.text((6, 4 + 16 * i), ln, fill=(0, 0, 0))
    out.save(os.path.join(a.out, a.name + ".png"))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
