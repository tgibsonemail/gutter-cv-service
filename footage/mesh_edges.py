#!/usr/bin/env python3
"""PROOF OF CONCEPT (not wired into the model). Find gutter runs from Google's 3D model (the Google Earth view).
Idea: every roof edge with a wall dropping away below it is a candidate.
  level edge + roof drains toward it  = eave  -> gutter   (unless it lands on pitched roof below: rule 4, left off)
  edge that climbs                    = rake / gable -> no gutter
  edge of a low slope cover           = patio cover -> left off (rule 3)
Works the same for dormers and upper sections because it never splits the roof into planes.
Usage: python3 mesh_edges.py <name>     needs results_3d/<name>_mesh.npz and results_v2/<name>_gmap.json
Writes results_3d/<name>_edges.png and <name>_edges.json"""
import sys, os, json, math, numpy as np, cv2
from scipy import ndimage as ndi
from skimage import morphology, draw
from pyproj import Transformer
import eave_planes as P, eave_test as E

HERE = os.path.dirname(os.path.abspath(__file__))
PX = 0.1; RAD = 32.0; M2FT = 3.28084
DROP_M = 1.0            # a height fall of this much within half a meter = a wall below the edge
ROOF_MIN_H = 2.0        # the high side must be at least this far above the ground
RAKE_SLOPE = 0.2        # edge climbing faster than this = rake (gable)
EAVE_DOT = 0.5          # roof drains toward the edge
COVER_PITCH = math.degrees(math.atan(2 / 12.0))
ROUGH_M = 0.18          # surface this lumpy over about 1.5 m = tree, not roof
FOOT_BUF_M = 1.5        # edges must sit within this distance of the map view building shape
MIN_RUN_FT = 3.0
GABLE_MIN_FT = 8.0      # a rake stretch shorter than this is not counted as a gable end for returns
RETURN_FT = 3.0         # gutter wraps about this far at each side of a gable end
TREE_GREEN_SHARE = 0.4
PEAK_SIDE_MIN_PX = 12   # each side of a gable peak must be at least 1.2 m long
PEAK_MAX_RMS_M = 0.25   # each side must be this close to a straight line
PEAK_MIN_RISE_M = 0.6   # and climb at least this much
JOG_MIN_SIDE_M = 0.9    # outline sides shorter than about 3 ft are wiggles, not jogs
JOG_SIMPLIFY_M = 0.5  # this share of green around a spot = tree

def rasterize(name):
    z = np.load(os.path.join(HERE, "results_3d", name + "_mesh.npz"))
    return rasterize_tri(z["tri"], z["col"])


def rasterize_tri(T, COL):
    """3D model triangles (east, north, up meters) -> top down height map Z and colour map C on a 0.1 m grid."""
    T = np.asarray(T, np.float64)
    N = int(2 * RAD / PX)
    Z = np.full((N, N), -np.inf); C = np.zeros((N, N, 3), np.uint8)
    cols = (T[:, :, 0] + RAD) / PX; rows = (RAD - T[:, :, 1]) / PX
    for i in range(len(T)):
        c, r, h = cols[i], rows[i], T[i, :, 2]
        c0, c1 = int(max(np.floor(c.min()), 0)), int(min(np.ceil(c.max()), N - 1))
        r0, r1 = int(max(np.floor(r.min()), 0)), int(min(np.ceil(r.max()), N - 1))
        if c1 < c0 or r1 < r0:
            continue
        den = (r[1] - r[2]) * (c[0] - c[2]) + (c[2] - c[1]) * (r[0] - r[2])
        if abs(den) < 1e-9:
            continue
        gx, gy = np.meshgrid(np.arange(c0, c1 + 1) + 0.0, np.arange(r0, r1 + 1) + 0.0)
        w0 = ((r[1] - r[2]) * (gx - c[2]) + (c[2] - c[1]) * (gy - r[2])) / den
        w1 = ((r[2] - r[0]) * (gx - c[2]) + (c[0] - c[2]) * (gy - r[2])) / den
        w2 = 1 - w0 - w1
        ins = (w0 >= -0.02) & (w1 >= -0.02) & (w2 >= -0.02)
        if not ins.any():
            continue
        hz = w0 * h[0] + w1 * h[1] + w2 * h[2]
        sub = Z[r0:r1 + 1, c0:c1 + 1]; subc = C[r0:r1 + 1, c0:c1 + 1]
        up = ins & (hz > sub)
        sub[up] = hz[up]; subc[up] = COL[i]
    Z[~np.isfinite(Z)] = np.nan
    return Z, C

def footprint_mask(name):
    """Map view building shape, moved from the Solar grid into the 3D model grid."""
    d = json.load(open(os.path.join(HERE, "results_v2", name + "_gmap.json")))
    key = E.read_key(E.DEFAULT_ENV)
    D = P.fetch(d["lat"], d["lng"], key)
    fp, info, _ = P.gmap_footprint(d["lat"], d["lng"], D, key)
    return footprint_from(D, fp)


def footprint_from(D, fp):
    """Move the map view building shape (row, col in the Solar grid) into the 3D model grid."""
    cr, cc = D["center"]
    tr = Transformer.from_crs(D["epsg"], 4326, always_xy=True)
    clng, clat = tr.transform(D["x0"] + cc * D["px"], D["y0"] - cr * D["px"])
    N = int(2 * RAD / PX); pts = []
    for r, c in fp:
        lng, lat = tr.transform(D["x0"] + c * D["px"], D["y0"] - r * D["px"])
        e = math.radians(lng - clng) * 6378137.0 * math.cos(math.radians(clat))
        n = math.radians(lat - clat) * 6356752.0
        pts.append(((RAD - n) / PX, (e + RAD) / PX))
    m = np.zeros((N, N), bool)
    rr, cc_ = draw.polygon([p[0] for p in pts], [p[1] for p in pts], (N, N)); m[rr, cc_] = True
    per = sum(math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(pts, pts[1:] + pts[:1])) * PX * M2FT
    return m, per, pts

def main(name):
    Z, C = rasterize(name)
    foot, foot_per, fpts = footprint_mask(name)
    res = measure(Z, C, foot, foot_per, fpts, name, png_path=os.path.join(HERE, "results_3d", name + "_edges.png"))
    json.dump(res, open(os.path.join(HERE, "results_3d", name + "_edges.json"), "w"), indent=1)
    print(json.dumps(res))


def measure(Z, C, foot, foot_per, fpts, name="house", png_path=None):
    """The measuring step. Z, C from rasterize_tri; foot, foot_per, fpts from footprint_from. Returns the result dict."""
    N = Z.shape[0]
    near = ndi.binary_dilation(foot, iterations=int(FOOT_BUF_M / PX))
    Zf = np.where(np.isfinite(Z), Z, np.nanmin(Z))
    ground = float(np.nanpercentile(Z[ndi.binary_dilation(foot, iterations=80) & ~ndi.binary_dilation(foot, iterations=20)], 10))
    Hh = Zf - ground
    # smooth surface slope, ignoring pixels that sit on a step
    rng = ndi.maximum_filter(Zf, 5) - ndi.minimum_filter(Zf, 5)
    flat = rng < 0.6
    gr, gc = np.gradient(Zf, PX)
    gr_s = E.norm_gauss(np.where(flat, gr, 0.0), flat, 3.0); gc_s = E.norm_gauss(np.where(flat, gc, 0.0), flat, 3.0)
    pitch = np.degrees(np.arctan(np.hypot(gr_s, gc_s)))
    # lumpiness: how far the surface is from its own local average plane
    resid = np.abs(Zf - ndi.uniform_filter(Zf, 15))
    rough = ndi.uniform_filter(np.where(flat, np.abs(gr - gr_s) + np.abs(gc - gc_s), 0.0), 15)
    # trees: the 3D model colours them green. Share of green pixels within about 0.75 m.
    Cf = C.astype(float)
    green = (Cf[..., 1] > Cf[..., 0] * 1.04) & (Cf[..., 1] > Cf[..., 2] * 1.0)
    tree = ndi.uniform_filter(green.astype(float), 15) > TREE_GREEN_SHARE
    # ---- drop edges: high side pixels with a fall of DROP_M within 0.5 m ----
    dirs = [(math.cos(a), math.sin(a)) for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)]
    best = np.zeros((N, N)); bdir = np.zeros((N, N), int)
    for k, (dr, dc) in enumerate(dirs):
        sh = ndi.shift(Zf, (-5 * dr, -5 * dc), order=0, mode="nearest")     # value at p + 5*dir
        dp = Zf - sh
        up = dp > best
        best[up] = dp[up]; bdir[up] = k
    edge = (best > DROP_M) & (Hh > ROOF_MIN_H) & near
    # keep only the top lip of the wall: the pixel one step further out must already be falling
    sk = morphology.skeletonize(morphology.binary_closing(edge, morphology.disk(1)))
    lab, n = ndi.label(sk, structure=np.ones((3, 3)))
    out = []
    global DBG
    DBG = []
    rakelike = np.zeros((N, N), bool)
    tang = np.zeros((N, N, 2))
    cls = np.zeros((N, N), np.uint8)   # 1 eave gutter, 2 rake, 3 left off onto roof, 4 cover edge, 5 other, 6 tree
    for j in range(1, n + 1):
        rr, cc = np.nonzero(lab == j)
        if len(rr) < 8:
            continue
        for r, c in zip(rr, cc):
            w = (np.abs(rr - r) <= 7) & (np.abs(cc - c) <= 7)
            if w.sum() < 6:
                continue
            pr, pc = rr[w].astype(float), cc[w].astype(float)
            q = np.c_[pr - pr.mean(), pc - pc.mean()]
            _, s, vt = np.linalg.svd(q, full_matrices=False)
            t = vt[0]
            nrm = np.array([-t[1], t[0]])
            dk = np.array(dirs[bdir[r, c]])
            if nrm @ dk < 0:
                nrm = -nrm
            # edge slope: height along the lip
            u = q @ t
            zz = Zf[rr[w], cc[w]]
            if (u.max() - u.min()) < 6:
                continue
            slope = abs(np.polyfit(u * PX, zz, 1)[0])
            # roof surface 0.8 m inside
            ir, ic = int(round(r - 8 * nrm[0])), int(round(c - 8 * nrm[1]))
            orr, oc = int(round(r + 8 * nrm[0])), int(round(c + 8 * nrm[1]))
            if not (0 <= ir < N and 0 <= ic < N and 0 <= orr < N and 0 <= oc < N):
                continue
            down = -np.array([gr_s[ir, ic], gc_s[ir, ic]])
            dn = np.hypot(*down)
            dot = float(down @ nrm / dn) if dn > 1e-6 else 0.0
            pin = pitch[ir, ic]
            hl = Hh[orr, oc]; pl = pitch[orr, oc]
            tree_out = False
            for off in (6, 10, 15, 20):
                a_, b_ = int(round(r + off * nrm[0])), int(round(c + off * nrm[1]))
                if 0 <= a_ < N and 0 <= b_ < N and tree[a_, b_]:
                    tree_out = True          # green just outside the edge: a tree is against it
            tang[r, c] = t
            if slope > RAKE_SLOPE and dot < 0.75:
                rakelike[r, c] = True
            if tree[ir, ic] or tree[r, c]:
                k = 6          # the high side itself is a tree: not a roof edge
            elif tree_out and (slope > RAKE_SLOPE and dot < 0.75):
                k = 7          # a tree is against this edge, so we cannot tell if it is a gable: count it as gutter, flagged (rule 7)
            elif pin < COVER_PITCH:
                k = 4
            elif slope > RAKE_SLOPE and dot < 0.75:
                k = 2
            elif dot > EAVE_DOT:
                k = 3 if (hl > ROOF_MIN_H and pl >= COVER_PITCH and not tree[orr, oc]) else 1
            else:
                k = 5
            if os.environ.get("MDBG") and k in (1, 3) :
                prof = []
                for off in (4, 8, 12, 16, 22, 30):
                    a_, b_ = int(round(r + off * nrm[0])), int(round(c + off * nrm[1]))
                    if 0 <= a_ < N and 0 <= b_ < N:
                        prof.append("%.1f/%.0f" % (Hh[a_, b_], pitch[a_, b_]))
                DBG.append((j, r, c, k, round(float(Hh[r, c]), 1), round(float(pin)), " ".join(prof)))
            cls[r, c] = k
            out.append((j, r, c, k, 1.0 / max(abs(t[0]), abs(t[1]), 1e-6) * PX))
    bnd = foot & ~ndi.binary_erosion(foot)
    dist_out = ndi.distance_transform_edt(~bnd) * PX        # meters from the map view outline
    if os.environ.get("MDBG"):
        seenj = {}
        for (j, r, c, k, h, pin, prof) in DBG:
            if dist_out[r, c] > 1.5:
                seenj.setdefault((j, k), []).append((r, c, h, pin, prof))
        for (j, k), v in seenj.items():
            m_ = v[len(v) // 2]
            print("INTERIOR chain", j, "class", k, "px", len(v), "mid rc", m_[0], m_[1], "lip h", m_[2], "roof pitch", m_[3], "| outside h/pitch at 0.4,0.8,1.2,1.6,2.2,3.0 m:", m_[4], file=sys.stderr)
    # ---- whole run rule (Trey, 2026-11-18, from 9505 Marbella): an upper eave is judged as a whole run.
    # If part of a straight run hangs over open air and part over roof below, the WHOLE run gets gutter ("you can't do only half").
    # Only a run that sits entirely over roof below is left off. ----
    whole_run_ft = 0.0
    for _pass in range(3):
        l3, n3 = ndi.label(cls == 3, structure=np.ones((3, 3)))
        l1, n1 = ndi.label(cls == 1, structure=np.ones((3, 3)))
        len1 = {}
        for o in out:
            if cls[o[1], o[2]] == 1:
                len1[l1[o[1], o[2]]] = len1.get(l1[o[1], o[2]], 0.0) + o[4] * M2FT
        changed = False
        for i in range(1, n3 + 1):
            rr_, cc_ = np.nonzero(l3 == i)
            if len(rr_) < 5:
                continue
            tv = tang[rr_, cc_]
            tv = tv * np.sign(tv @ tv[0])[:, None]
            tm = tv.mean(axis=0); tm = tm / max(np.hypot(*tm), 1e-9)
            u = (rr_ - rr_.mean()) * tm[0] + (cc_ - cc_.mean()) * tm[1]
            joined = False
            for e_i in (int(np.argmin(u)), int(np.argmax(u))):
                er, ec = rr_[e_i], cc_[e_i]
                r0_, r1_, c0_, c1_ = max(er - 7, 0), min(er + 8, N), max(ec - 7, 0), min(ec + 8, N)
                nb = np.argwhere(cls[r0_:r1_, c0_:c1_] == 1)
                for a_, b_ in nb:
                    pr_, pc_ = r0_ + a_, c0_ + b_
                    if abs(float(tang[pr_, pc_] @ tm)) > 0.9 and len1.get(l1[pr_, pc_], 0.0) >= MIN_RUN_FT:
                        joined = True
                        break
                if joined:
                    break
            if joined:
                whole_run_ft += sum(o[4] for o in out if l3[o[1], o[2]] == i) * M2FT
                cls[rr_, cc_] = 1
                changed = True
        if not changed:
            break
    # ---- clean peak exception (Trey, 2026-11-18): a tree next to an edge normally means "cannot tell, count it".
    # But if the edge still shows a tidy gable shape (straight climb, a point, straight fall) trust it as a gable. ----
    peak_fixed = 0
    cand = (np.isin(cls, (2, 7)) | ((cls == 6) & rakelike)) & (dist_out <= 1.5)
    plab, pn = ndi.label(ndi.binary_dilation(cand, iterations=3))
    for i in range(1, pn + 1):
        rr_, cc_ = np.nonzero(cand & (plab == i))
        if len(rr_) < 2 * PEAK_SIDE_MIN_PX or not np.isin(cls[rr_, cc_], (6, 7)).any():
            continue
        q = np.c_[rr_ - rr_.mean(), cc_ - cc_.mean()].astype(float)
        _, _, vt = np.linalg.svd(q, full_matrices=False)
        u = (q @ vt[0]) * PX
        o_ = np.argsort(u); u = u[o_]; zz = ndi.median_filter(Zf[rr_[o_], cc_[o_]], 5)
        ip = int(np.argmax(zz))
        if ip < PEAK_SIDE_MIN_PX or len(u) - ip - 1 < PEAK_SIDE_MIN_PX:
            continue
        okp = True
        for sl, sgn in ((slice(0, ip + 1), 1.0), (slice(ip, len(u)), -1.0)):
            cf = np.polyfit(u[sl], zz[sl], 1)
            rms = float(np.sqrt(np.mean((np.polyval(cf, u[sl]) - zz[sl]) ** 2)))
            rise = float(zz[ip] - (zz[sl][0] if sgn > 0 else zz[sl][-1]))
            if cf[0] * sgn < RAKE_SLOPE or rms > PEAK_MAX_RMS_M or rise < PEAK_MIN_RISE_M:
                okp = False
        if okp:
            m7 = np.isin(cls[rr_, cc_], (6, 7))
            cls[rr_[m7], cc_[m7]] = 2
            peak_fixed += 1
    tot = {}
    names = {1: "eave_gutter", 2: "rake_gable", 3: "left_off_onto_roof", 4: "cover_edge", 5: "other_edge", 6: "tree", 7: "hidden_by_tree_counted"}
    # drop short runs
    for k in names:
        m = cls == k
        lb, nn = ndi.label(m, structure=np.ones((3, 3)))
        ft = 0.0
        for i in range(1, nn + 1):
            L = sum(o[4] for o in out if lb[o[1], o[2]] == i) * M2FT
            if L >= MIN_RUN_FT:
                ft += L
            else:
                cls[lb == i] = 0
        tot[names[k]] = round(ft)
        tot[names[k] + "_on_outline"] = round(sum(o[4] for o in out if cls[o[1], o[2]] == k and dist_out[o[1], o[2]] <= 1.5) * M2FT)
        tot[names[k] + "_interior"] = round(sum(o[4] for o in out if cls[o[1], o[2]] == k and dist_out[o[1], o[2]] > 1.5) * M2FT)
    # returns (Trey, 2026-11-18): where a gutter run stops at a gable it wraps about 3 ft. Count gable ends seen on the outline, 2 returns each.
    rk = (cls == 2) & (dist_out <= 1.5)
    glab, gn = ndi.label(ndi.binary_dilation(rk, iterations=6))
    gables = 0
    for i in range(1, gn + 1):
        L = sum(o[4] for o in out if cls[o[1], o[2]] == 2 and glab[o[1], o[2]] == i) * M2FT
        if L >= GABLE_MIN_FT:
            gables += 1
    returns_ft = gables * 2 * RETURN_FT
    # jogs: clean the map outline first so small wiggles are not counted as corners
    poly = np.array([[p[1], p[0]] for p in fpts], np.float32).reshape(-1, 1, 2)
    ap = cv2.approxPolyDP(poly, JOG_SIMPLIFY_M / PX, True).reshape(-1, 2)
    changed = True
    while changed and len(ap) > 4:
        changed = False
        for i in range(len(ap)):
            a_, b_ = ap[i], ap[(i + 1) % len(ap)]
            if np.hypot(*(a_ - b_)) * PX < JOG_MIN_SIDE_M:
                ap = np.delete(ap, i, axis=0); changed = True
                break
    corners = 0; turn = 0.0
    for i in range(len(ap)):
        v1 = ap[i] - ap[i - 1]; v2 = ap[(i + 1) % len(ap)] - ap[i]
        ang = abs(math.degrees(math.atan2(v1[0] * v2[1] - v1[1] * v2[0], v1[0] * v2[0] + v1[1] * v2[1])))
        if ang > 30:
            corners += 1
        turn += ang
    jogs = max(turn / 90.0 - 4, 0) / 2.0        # a square corner jog = 2 right angles; a 45 degree bay also adds up to 2
    jog_returns_ft = jogs * RETURN_FT
    seen = sum(tot[names[k] + "_on_outline"] for k in names)
    combo = foot_per - tot["rake_gable_on_outline"] + tot["eave_gutter_interior"]
    res = dict(name=name, map_shape_perimeter_ft=round(foot_per), outline_seen_in_3d_ft=seen, ground=round(ground, 1),
               combo_ft=round(combo), gable_ends_seen=gables, returns_ft=returns_ft, combo_with_returns_ft=round(combo + returns_ft),
               clean_corners=corners, jogs=round(jogs, 1), jog_returns_ft=round(jog_returns_ft, 1), peaks_trusted_despite_tree=peak_fixed,
               whole_run_rule_added_ft=round(whole_run_ft),
               final_ft=round(combo + returns_ft + jog_returns_ft), combo_note="map outline - rakes seen on outline + interior eaves (dormers, upper sections)", totals_ft=tot)
    # typical edge height above the ground for each kind of edge (tells one story from two)
    res["edge_height_ft"] = {names[k]: (round(float(np.median(Hh[cls == k])) * M2FT, 1) if (cls == k).sum() > 10 else None) for k in names}
    if png_path is None:
        return res
    img = C[:, :, ::-1].copy()
    colr = {1: (0, 220, 0), 2: (0, 0, 255), 3: (0, 165, 255), 4: (255, 255, 255), 5: (0, 255, 255), 6: (255, 0, 255), 7: (160, 255, 160)}
    big = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
    for k, cl in colr.items():
        for r, c in zip(*np.nonzero(cls == k)):
            cv2.circle(big, (int(c * 3 + 1), int(r * 3 + 1)), 3, cl, -1)
    pp = np.array([[p[1] * 3, p[0] * 3] for p in fpts], np.int32)
    cv2.polylines(big, [pp], True, (255, 200, 0), 1)
    rr, cc = np.nonzero(near); m = 60
    r0, r1, c0, c1 = max(rr.min() - m, 0) * 3, min(rr.max() + m, N) * 3, max(cc.min() - m, 0) * 3, min(cc.max() + m, N) * 3
    cv2.imwrite(png_path, big[r0:r1, c0:c1])
    return res

if __name__ == "__main__":
    main(sys.argv[1])
