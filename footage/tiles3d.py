#!/usr/bin/env python3
"""Pull Google's 3D model (Map Tiles API, Photorealistic 3D Tiles: the same data as the Google Earth view) around a point.
Returns triangles in local east / north / up meters. No drawing here, so it is fast.
Billing note: one "root" request per address is the billable event; the tile requests under it ride on that session."""
import io, json, math, struct, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
from pyproj import Transformer
import netlimits as NL
try:
    import DracoPy          # only needed if Google ever serves Draco compressed tiles; today it does not
except Exception:
    DracoPy = None

BASE = "https://tile.googleapis.com"
RADIUS_M = 32.0

def fetch(url):
    return NL.read(url)       # 0.1.2: time limit + one retry on 5xx/timeout (netlimits.py)


def obb_hits_sphere(box, c, r):
    ctr = np.array(box[0:3]); ax = [np.array(box[3:6]), np.array(box[6:9]), np.array(box[9:12])]
    d = c - ctr
    q = ctr.copy()
    for a in ax:
        L = np.linalg.norm(a)
        if L < 1e-9:
            continue
        u = a / L
        q = q + np.clip(np.dot(d, u), -L, L) * u
    return np.linalg.norm(c - q) <= r


def glb_meshes(data):
    """Return list of (points Nx3 float64 in glTF coords with node matrices applied, faces Mx3, uv Nx2 or None, image or None)."""
    magic, ver, length = struct.unpack("<4sII", data[:12])
    off = 12
    js = None; binc = None
    while off < length:
        clen, ctype = struct.unpack("<I4s", data[off:off + 8])
        chunk = data[off + 8: off + 8 + clen]
        if ctype == b"JSON":
            js = json.loads(chunk)
        elif ctype[:3] == b"BIN":
            binc = chunk
        off += 8 + clen
    out = []
    bvs = js.get("bufferViews", [])

    def bv(i):
        v = bvs[i]
        o = v.get("byteOffset", 0)
        return binc[o:o + v["byteLength"]]
    images = {}

    def image_for(mat_i):
        try:
            tex_i = js["materials"][mat_i]["pbrMetallicRoughness"]["baseColorTexture"]["index"]
            src = js["textures"][tex_i]["source"]
        except Exception:
            return None
        if src not in images:
            im = js["images"][src]
            try:
                images[src] = np.array(Image.open(io.BytesIO(bv(im["bufferView"]))).convert("RGB"))
            except Exception:
                images[src] = None
        return images[src]

    CT = {5120: (np.int8, 127.0), 5121: (np.uint8, 255.0), 5122: (np.int16, 32767.0), 5123: (np.uint16, 65535.0),
          5125: (np.uint32, 1.0), 5126: (np.float32, 1.0)}
    NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}

    def accessor(i):
        ac = js["accessors"][i]
        v = bvs[ac["bufferView"]]
        dt, nrm = CT[ac["componentType"]]
        ncomp = NC[ac["type"]]
        start = v.get("byteOffset", 0) + ac.get("byteOffset", 0)
        stride = v.get("byteStride", 0)
        item = np.dtype(dt).itemsize * ncomp
        if stride and stride != item:
            raw = np.frombuffer(binc, np.uint8, count=stride * ac["count"], offset=start).reshape(ac["count"], stride)[:, :item]
            arr = np.frombuffer(raw.tobytes(), dt).reshape(ac["count"], ncomp)
        else:
            arr = np.frombuffer(binc, dt, count=ac["count"] * ncomp, offset=start).reshape(ac["count"], ncomp)
        if ac.get("normalized"):
            arr = arr.astype(float) / nrm
        return arr if ncomp > 1 else arr.reshape(-1)

    def node_mats(ni, parent):
        n = js["nodes"][ni]
        M = np.eye(4)
        if "matrix" in n:
            M = np.array(n["matrix"], float).reshape(4, 4).T
        else:
            T = np.eye(4); R = np.eye(4); S = np.eye(4)
            if "translation" in n:
                T[:3, 3] = n["translation"]
            if "rotation" in n:
                x, y, z, w = n["rotation"]
                R[:3, :3] = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
            if "scale" in n:
                S[0, 0], S[1, 1], S[2, 2] = n["scale"]
            M = T @ R @ S
        M = parent @ M
        if "mesh" in n:
            for prim in js["meshes"][n["mesh"]]["primitives"]:
                ext = prim.get("extensions", {}).get("KHR_draco_mesh_compression")
                uv = None
                if ext:
                    dm = DracoPy.decode(bytes(bv(ext["bufferView"])))
                    pts = np.asarray(dm.points, float).reshape(-1, 3)
                    faces = np.asarray(dm.faces).reshape(-1, 3)
                    try:
                        tc = np.asarray(dm.tex_coord, float)
                        if tc.size:
                            uv = tc.reshape(-1, 2)
                    except Exception:
                        uv = None
                else:
                    at = prim.get("attributes", {})
                    if "POSITION" not in at or "indices" not in prim or prim.get("mode", 4) != 4:
                        continue
                    pts = accessor(at["POSITION"]).astype(float)
                    faces = accessor(prim["indices"]).reshape(-1, 3).astype(np.int64)
                    if "TEXCOORD_0" in at:
                        uv = accessor(at["TEXCOORD_0"]).astype(float)
                ph = np.c_[pts, np.ones(len(pts))] @ M.T
                img = image_for(prim["material"]) if "material" in prim else None
                out.append((ph[:, :3], faces, uv, img))
        for c in n.get("children", []):
            node_mats(c, M)
    sc = js["scenes"][js.get("scene", 0)]
    for ni in sc["nodes"]:
        node_mats(ni, np.eye(4))
    return out



def fetch_mesh(lat, lng, h_msl, key, radius_m=RADIUS_M, workers=8):
    """lat/lng = building center (use the Solar API building center so both data sets share one origin).
    h_msl = rough roof height above sea level (from Solar roofSegmentStats). Returns (T, COL, n_requests):
    T (n,3,3) east/north/up meters with the 5th percentile height moved to 0, COL (n,3) RGB per triangle."""
    to_ecef = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    c = np.array(to_ecef.transform(lng, lat, h_msl - 27.0))      # rough geoid offset for north Texas; only used to pick tiles
    la, lo = math.radians(lat), math.radians(lng)
    east = np.array([-math.sin(lo), math.cos(lo), 0.0])
    north = np.array([-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)])
    up = np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])
    session = {"q": ""}
    count = {"req": 1}

    def full(uri):
        u = uri if uri.startswith("http") else BASE + uri
        if "session=" in u:
            session["q"] = u.split("session=")[1].split("&")[0]
        elif session["q"]:
            u += ("&" if "?" in u else "?") + "session=" + session["q"]
        u += ("&" if "?" in u else "?") + "key=" + key
        return u

    def collect(tile, uris):
        box = tile.get("boundingVolume", {}).get("box")
        if box and not obb_hits_sphere(box, c, radius_m + 10):
            return
        kids = tile.get("children", [])
        if kids:
            for k in kids:
                collect(k, uris)
            return
        uri = tile.get("content", {}).get("uri")
        if uri:
            uris.append(uri)

    leaves = []
    root = json.loads(fetch(BASE + "/v1/3dtiles/root.json?key=" + key))
    level = []
    collect(root["root"], level)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        while level:
            urls = [full(u) for u in level]            # also picks up the session token
            count["req"] += len(urls)
            datas = list(ex.map(fetch, urls))
            level = []
            for data in datas:
                if data[:4] == b"glTF":
                    leaves.append(data)
                else:
                    collect(json.loads(data)["root"], level)
    P, C = [], []
    yup = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)     # glTF Y up -> Z up
    for data in leaves:
        for pts, faces, uv, img in glb_meshes(data):
            w = pts @ yup.T
            if np.linalg.norm(w.mean(axis=0) - c) > 5000:
                w = pts
            rel = w - c
            enu = np.c_[rel @ east, rel @ north, rel @ up]
            tri = enu[faces]
            cen = tri.mean(axis=1)
            keep = np.hypot(cen[:, 0], cen[:, 1]) < radius_m
            if not keep.any():
                continue
            tri = tri[keep]
            if uv is not None and img is not None:
                uvc = uv[faces[keep]].mean(axis=1)
                hh, ww = img.shape[:2]
                x = np.clip((uvc[:, 0] * ww).astype(int), 0, ww - 1)
                y = np.clip((uvc[:, 1] * hh).astype(int), 0, hh - 1)
                col = img[y, x]
            else:
                col = np.full((len(tri), 3), 160, np.uint8)
            P.append(tri); C.append(col)
    if not P:
        return None, None, count["req"]
    T = np.concatenate(P); COL = np.concatenate(C)
    T = T - np.array([0, 0, float(np.percentile(T[:, :, 2], 5))])
    return T, COL, count["req"]
