#!/usr/bin/env python3
"""
EasyGutters gutter footage engine. ONE COMMAND: address in, gutter footage + range + flags out.

    python3 estimate.py --address "9505 Marbella Dr, Fort Worth, TX 76126"
    python3 estimate.py --address "..." --png check.png      (also saves a picture of what was counted)
Every run from the command line is added to the history (history/estimates.jsonl) unless --no-log is given.
When calling estimate() from Python, save it yourself with history.log_estimate(result, address_input=...).

Or from Python:
    from estimate import estimate
    result = estimate(address="...", key=GOOGLE_KEY)

The Google key is read from the environment variable GOOGLE_MAPS_API_KEY, or from an env file given with --env.
It is never written to disk or printed. Nothing fetched from Google is saved unless you pass --png.

How the number is built (see HANDOFF.md for the full story):
    footage = map view outline of the house
              - gable ends the 3D model can see on that outline
              + eaves the 3D model sees inside the outline (dormers, upper sections not over roof)
              + 3 ft at each side of every gable end (returns)
              + 3 ft at every jog in the roofline (returns)
"""
import argparse, json, os, sys, time, urllib.parse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import eave_test as E
import eave_planes as P
import mesh_edges as M
import tiles3d

ENGINE_VERSION = "0.1.1 (2026-11-18)"   # 0.1.1: speed only (footprint nudge), numbers identical to 0.1
RANGE_PCT = 10.0             # Trey's decision 2026-11-18: quote shows plus or minus 10% of the number
# ---- flag thresholds: set by judgment from eleven houses, NOT statistically proven ----
PLANE_SHARE_LOW = 0.60       # share of the roof that resolves into clean planes in the Solar height data. Good houses 0.74 to 0.84, the one bad house 0.48
SEEN_3D_LOW = 0.60           # the 3D model should see at least this share of the outline
TREE_SHARE_HIGH = 0.25       # trees against more than this share of the outline
UPPER_LEFT_OFF_NOTE_FT = 30  # tell the user when this much upper roof was left off
CORNERS_TESTED_MAX = 21      # most corners on any house where the 3 ft jog rule tested well
SECOND_STRUCT_MIN_SQFT = 100


def _key(env_path=None):
    k = os.environ.get("GOOGLE_MAPS_API_KEY")
    if k:
        return k.strip()
    if env_path:
        return E.read_key(env_path)
    raise RuntimeError("No Google key: set GOOGLE_MAPS_API_KEY or pass --env <file>")


def _fail(status, message, **extra):
    d = dict(status=status, message=message, footage_ft=None, range_low_ft=None, range_high_ft=None,
             confidence="none", flags=[dict(code=status, message=message)], engine_version=ENGINE_VERSION)
    d.update(extra)
    return d


def estimate(address=None, lat=None, lng=None, key=None, env_path=None, png_path=None, range_pct=RANGE_PCT):
    key = key or _key(env_path)
    T0 = time.time(); tm = {}
    formatted, loc_type = None, None
    # 1. address -> point
    if address:
        t = time.time()
        gj = E.get_json("https://maps.googleapis.com/maps/api/geocode/json?" + urllib.parse.urlencode({"address": address, "key": key}))
        if gj.get("status") != "OK":
            return _fail("address_not_found", "Google could not find this address (%s)" % gj.get("status"), address=address)
        g0 = gj["results"][0]
        lat, lng = g0["geometry"]["location"]["lat"], g0["geometry"]["location"]["lng"]
        formatted, loc_type = g0.get("formatted_address"), g0["geometry"].get("location_type")
        tm["geocode"] = round(time.time() - t, 1)
    # 2. which building (Solar API). No roof data here = cannot measure.
    t = time.time()
    try:
        bi = P.building_insights(lat, lng, key)
    except SystemExit as e:
        return _fail("no_roof_data", "Google has no roof data for this address (%s)" % str(e)[:120], address=formatted or address, lat=lat, lng=lng)
    tm["building"] = round(time.time() - t, 1)
    clat, clng = bi["center"]["latitude"], bi["center"]["longitude"]
    h_msl = bi.get("solarPotential", {}).get("roofSegmentStats", [{}])[0].get("planeHeightAtCenterMeters", 180.0)

    # 3. two downloads side by side: (a) Solar height data + map view outline, (b) the 3D model
    def solar_job():
        t = time.time()
        D = P.fetch(lat, lng, key, bi=bi)
        fp, info, _ = P.gmap_footprint(lat, lng, D, key)
        q = {}
        try:
            A = P.analyze(D, poly_override=fp) if fp is not None else P.analyze(D)
            q["plane_share"] = round(float((A["cat"] > 0).sum()) / max(float(A["R"].sum()), 1.0), 2)
            q["patio_cover_sqft"] = round(float(A["cover"].sum()) * D["px"] ** 2 * 10.7639)
            ar = {}
            for p in A["planes"]:
                ar[p["struct"]] = ar.get(p["struct"], 0.0) + p["area_m2"] * 10.7639
            srt = sorted(ar.values(), reverse=True)
            q["other_structures_sqft"] = [round(v) for v in srt[1:] if v >= SECOND_STRUCT_MIN_SQFT]
        except BaseException as e:           # the quality check must never sink the estimate
            q["quality_check_error"] = str(e)[:120]
        return D, fp, q, round(time.time() - t, 1)

    def mesh_job():
        t = time.time()
        T, COL, nreq = tiles3d.fetch_mesh(clat, clng, h_msl, key)
        return T, COL, nreq, round(time.time() - t, 1)

    with ThreadPoolExecutor(max_workers=2) as ex:
        f1, f2 = ex.submit(solar_job), ex.submit(mesh_job)
        try:
            D, fp, q, tm["solar_and_outline"] = f1.result()
        except SystemExit as e:
            return _fail("no_roof_data", "Google roof data could not be read (%s)" % str(e)[:120], address=formatted or address, lat=lat, lng=lng)
        try:
            T, COL, nreq, tm["model_3d"] = f2.result()
        except Exception as e:
            return _fail("no_3d_model", "Google 3D model could not be read (%s)" % str(e)[:120], address=formatted or address, lat=lat, lng=lng)
    if fp is None:
        return _fail("no_map_shape", "Google's map view shows no building shape at this address", address=formatted or address, lat=lat, lng=lng)
    if T is None:
        return _fail("no_3d_model", "Google has no 3D model here", address=formatted or address, lat=lat, lng=lng)

    # 4. measure
    t = time.time()
    Z, C = M.rasterize_tri(T, COL)
    foot, foot_per, fpts = M.footprint_from(D, fp)
    res = M.measure(Z, C, foot, foot_per, fpts, name="house", png_path=png_path)
    tm["measure"] = round(time.time() - t, 1)
    tot = res["totals_ft"]
    ft = float(res["final_ft"])
    outline = float(res["map_shape_perimeter_ft"])

    # 5. flags
    flags = []; low = False

    def flag(code, msg, serious=False):
        nonlocal low
        flags.append(dict(code=code, message=msg, serious=serious))
        low = low or serious
    if loc_type and loc_type != "ROOFTOP":
        flag("address_not_exact", "Google matched the address only approximately (%s). The measured building may be the wrong one." % loc_type, True)
    ps = q.get("plane_share")
    if ps is not None and ps < PLANE_SHARE_LOW:
        flag("poor_height_data", "Roof height data is poor here (quality score %.2f, good houses score 0.74 or more). Usually heavy tree cover." % ps, True)
    seen = float(res["outline_seen_in_3d_ft"]) / max(outline, 1.0)
    if seen < SEEN_3D_LOW:
        flag("outline_mostly_unseen", "The 3D model could only see %d%% of the roof edge. Gables may have been missed." % round(100 * seen), True)
    tree_ft = tot["tree_on_outline"] + tot["hidden_by_tree_counted_on_outline"]
    if tree_ft / max(outline, 1.0) > TREE_SHARE_HIGH:
        flag("heavy_trees", "Trees are against about %d ft of the roof edge. Hidden edges are counted as gutter." % tree_ft, True)
    if D.get("quality") != "HIGH":
        flag("imagery_not_high", "Google imagery quality is %s for this address." % D.get("quality"))
    if tot["left_off_onto_roof"] >= UPPER_LEFT_OFF_NOTE_FT:
        flag("upper_roof_left_off", "%d ft of upper roof was left off because it sits entirely over roof below. If the installer gutters it, add that much." % tot["left_off_onto_roof"])
    if res["clean_corners"] > CORNERS_TESTED_MAX:
        flag("many_corners", "%d corners, more than the houses this was tested on. The 3 ft per jog allowance (%.0f ft here) may run high." % (res["clean_corners"], res["jog_returns_ft"]))
    if q.get("other_structures_sqft"):
        flag("other_structure", "Another roofed structure is on the lot (about %s sq ft of roof). It is NOT in this number: quote it as its own line." % ", ".join(str(v) for v in q["other_structures_sqft"]))
    if q.get("patio_cover_sqft", 0) > 0:
        flag("patio_cover", "A patio cover or other low slope roof (about %d sq ft) was found and left off." % q["patio_cover_sqft"])
    if "quality_check_error" in q:
        flag("quality_check_failed", "The data quality check could not run, so treat this number with care.")
    tm["total"] = round(time.time() - T0, 1)
    return dict(
        status="ok", address=formatted or address, lat=lat, lng=lng,
        footage_ft=round(ft), guard_footage_ft=round(ft),     # Trey 2026-11-18: guards get the same feet as the gutter runs, downspouts excluded
        range_pct=range_pct,
        range_low_ft=round(ft * (1 - range_pct / 100.0)), range_high_ft=round(ft * (1 + range_pct / 100.0)),
        confidence="low" if low else "normal", flags=flags,
        breakdown=dict(map_outline_ft=round(outline), gables_removed_ft=tot["rake_gable_on_outline"],
                       interior_eaves_added_ft=tot["eave_gutter_interior"], gable_ends=res["gable_ends_seen"],
                       gable_returns_ft=res["returns_ft"], jogs=res["jogs"], jog_returns_ft=res["jog_returns_ft"],
                       hidden_by_tree_counted_ft=tot["hidden_by_tree_counted_on_outline"], tree_against_edge_ft=tree_ft,
                       upper_roof_left_off_ft=tot["left_off_onto_roof"], whole_run_rule_added_ft=res["whole_run_rule_added_ft"],
                       patio_cover_edge_ft=tot["cover_edge"], gable_peaks_trusted_despite_tree=res["peaks_trusted_despite_tree"]),
        quality=dict(imagery_quality=D.get("quality"), imagery_date=D.get("date"), plane_share=ps,
                     outline_seen_in_3d_pct=round(100 * seen), edge_height_ft=res.get("edge_height_ft"),
                     geocode_match=loc_type, tile_requests=nreq),
        timings_s=tm, engine_version=ENGINE_VERSION,
        attribution="Source: Includes solar data from Google")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Gutter footage from an address")
    ap.add_argument("--address"); ap.add_argument("--lat", type=float); ap.add_argument("--lng", type=float)
    ap.add_argument("--env", help="env file holding GOOGLE_MAPS_API_KEY=... (if the environment variable is not set)")
    ap.add_argument("--png", help="save a picture of what was counted")
    ap.add_argument("--no-log", action="store_true", help="do not add this estimate to the history")
    a = ap.parse_args()
    if not a.address and a.lat is None:
        ap.error("give --address or --lat and --lng")
    try:
        out = estimate(address=a.address, lat=a.lat, lng=a.lng, env_path=a.env, png_path=a.png)
    except Exception as e:
        out = _fail("error", "Engine error: %s" % str(e)[:200])
    if not a.no_log:
        try:
            import history
            history.log_estimate(out, address_input=a.address, blind=True)
        except Exception as e:
            out.setdefault("flags", []).append(dict(code="history_not_saved", message="Could not write the history file: %s" % str(e)[:100], serious=False))
    print(json.dumps(out, indent=2))
