# Footage engine (v0.1, 2026-11-18)

Address in, gutter footage + plus or minus 10% range + flags out. Built in Trey's Business project and handed off
(see Easy Gutters outputs/2026-11-18_gutter-measure-engine/HANDOFF.md). Served by main.py at POST /footage.

- Needs the environment variable GOOGLE_MAPS_API_KEY (Geocoding, Solar, Maps Static, Map Tiles enabled). Server side only.
- Nothing fetched from Google is stored. The caller (quote engine) saves the returned numbers.
- Engine files here are copies of the tested package. Change them there first, rerun regression.py, bump ENGINE_VERSION, then copy here.
