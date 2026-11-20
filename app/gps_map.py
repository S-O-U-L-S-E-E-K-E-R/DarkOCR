#!/usr/bin/env python3
"""Put the GPS positions from a results CSV on a map.

Usage: ./map.sh results/<bucket>.csv [more.csv ...]
Writes results/<bucket>.map.html next to each CSV: one self-contained file, open it in a browser.

The map library (Leaflet 1.9.4) is embedded in the file. The only thing the browser fetches
from the internet is the background map (Esri by default; OpenStreetMap when chosen in the
map), so that provider sees which areas are viewed, not the photos or their positions.
"""

import collections
import csv
import html
import json
import os
import sys
import urllib.parse

APP_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_DIR = os.path.dirname(APP_DIR)  # the project folder
TEXT_PREVIEW = 300  # characters of image text shown per photo

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>__TITLE__</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>__LEAFLET_CSS__</style>
<style>
  html, body { height: 100%; margin: 0; font: 14px/1.4 system-ui, sans-serif; color: #1a1a1a; }
  #layout { display: flex; height: 100%; }
  #side { width: 340px; overflow-y: auto; border-right: 1px solid #ccc; background: #fafafa; }
  #side h1 { font-size: 16px; margin: 12px; }
  #side p.note { margin: 0 12px 10px; color: #555; font-size: 12px; }
  #filter { box-sizing: border-box; width: calc(100% - 24px); margin: 0 12px 8px; padding: 6px; }
  .place { padding: 8px 12px; border-top: 1px solid #e2e2e2; cursor: pointer; }
  .place:hover, .place.active { background: #e6f0ff; }
  .place b { display: block; }
  .place span { color: #555; font-size: 12px; }
  #map { flex: 1; }
  .photo { border-top: 1px solid #ddd; padding: 6px 0; overflow-wrap: anywhere; }
  .photo:first-child { border-top: 0; }
  .photo img { max-width: 220px; max-height: 160px; display: block; margin: 4px 0; }
  .photo .text { color: #444; font-size: 12px; }
  .leaflet-popup-content { max-height: 360px; overflow-y: auto; width: 250px !important; }
</style></head>
<body><div id="layout">
  <div id="side">
    <h1>__TITLE__</h1>
    <p class="note">__NOTE__</p>
    <input id="filter" placeholder="Filter by file name, camera, date, text">
    <div id="places"></div>
  </div>
  <div id="map"></div>
</div>
<script>__LEAFLET_JS__</script>
<script>
const places = __DATA__;
const map = L.map('map');
// Background maps, switchable at the top right. Esri serves tiles to a page opened from a
// local file; OpenStreetMap's own server refuses those, so it is only an extra choice.
const esri = 'https://server.arcgisonline.com/ArcGIS/rest/services/';
const backgrounds = {
  'Streets': L.tileLayer(esri + 'World_Street_Map/MapServer/tile/{z}/{y}/{x}', {
    maxZoom: 19, attribution: 'Tiles &copy; Esri'}),
  'Satellite': L.tileLayer(esri + 'World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    maxZoom: 19, attribution: 'Imagery &copy; Esri, Maxar, Earthstar Geographics'}),
  'OpenStreetMap': L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19, attribution: '&copy; OpenStreetMap contributors'}),
};
backgrounds['Streets'].addTo(map);
L.control.layers(backgrounds, null, {collapsed: false}).addTo(map);
L.control.scale().addTo(map);

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function photoHtml(photo) {
  const lines = ['<b>' + escapeHtml(photo.id) + '</b>'];
  if (photo.file) lines.push('<a href="' + photo.file + '" target="_blank"><img loading="lazy" src="' + photo.file + '"></a>');
  if (photo.taken) lines.push('Taken: ' + escapeHtml(photo.taken));
  if (photo.gpsTime) lines.push('GPS time: ' + escapeHtml(photo.gpsTime));
  if (photo.uploaded) lines.push('Uploaded: ' + escapeHtml(photo.uploaded));
  if (photo.camera) lines.push('Camera: ' + escapeHtml(photo.camera));
  if (photo.altitude) lines.push('Altitude: ' + escapeHtml(photo.altitude) + ' m');
  if (photo.copyOf) lines.push('Identical copy of: ' + escapeHtml(photo.copyOf));
  if (photo.text) lines.push('<span class="text">' + escapeHtml(photo.text) + '</span>');
  return '<div class="photo">' + lines.join('<br>') + '</div>';
}

const list = document.getElementById('places');
const entries = places.map(place => {
  const marker = L.circleMarker([place.lat, place.lon], {
    radius: 6 + 2 * Math.sqrt(place.photos.length), color: '#b3261e', weight: 2,
    fillColor: '#e5484d', fillOpacity: 0.6
  }).addTo(map);
  marker.bindPopup(() => '<b>' + place.lat + ', ' + place.lon + '</b> &middot; <a target="_blank" href="'
    + place.link + '">open in Google Maps</a>' + place.photos.map(photoHtml).join(''));
  const item = document.createElement('div');
  item.className = 'place';
  const dates = place.photos.map(p => p.taken).filter(Boolean).sort();
  item.innerHTML = '<b>' + place.lat + ', ' + place.lon + '</b><span>' + place.photos.length
    + (place.photos.length === 1 ? ' photo' : ' photos')
    + (dates.length ? ' &middot; ' + escapeHtml(dates[0]) : '') + '</span>';
  item.onclick = () => { map.setView([place.lat, place.lon], Math.max(map.getZoom(), 14)); marker.openPopup(); };
  marker.on('popupopen', () => { item.classList.add('active'); item.scrollIntoView({block: 'nearest'}); });
  marker.on('popupclose', () => item.classList.remove('active'));
  list.appendChild(item);
  const haystack = JSON.stringify(place).toLowerCase();
  return {marker, item, haystack};
});
map.fitBounds(L.latLngBounds(places.map(p => [p.lat, p.lon])).pad(0.1));

document.getElementById('filter').oninput = event => {
  const wanted = event.target.value.toLowerCase();
  for (const entry of entries) {
    const show = entry.haystack.includes(wanted);
    entry.item.style.display = show ? '' : 'none';
    if (show) entry.marker.addTo(map); else entry.marker.remove();
  }
};
</script></body></html>
"""


def build_map(csv_path):
    """Write <csv>.map.html. Return (path, photo count, place count), or None without GPS rows."""
    csv.field_size_limit(1 << 30)
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("gps_lat") and row.get("gps_lon")]
    if not rows:
        return None
    bucket = os.path.splitext(os.path.basename(csv_path))[0]
    out_path = os.path.splitext(csv_path)[0] + ".map.html"
    image_dir = os.path.join(HOME_DIR, "images", bucket)

    places = collections.OrderedDict()
    for row in sorted(rows, key=lambda row: (row["taken_at"] or "9", row["image_id"])):
        # An identical copy was not downloaded again: show the file of its original.
        local = os.path.join(image_dir, row["duplicate_of"] or row["image_id"])
        file_link = ""
        if os.path.isfile(local):
            relative = os.path.relpath(local, os.path.dirname(os.path.abspath(out_path)))
            file_link = urllib.parse.quote(relative.replace(os.sep, "/"))
        text = row["text"]
        photo = {
            "id": row["image_id"], "file": file_link, "taken": row["taken_at"],
            "gpsTime": row["gps_time_utc"], "uploaded": row["last_modified"],
            "camera": " ".join(filter(None, [row["camera_make"], row["camera_model"]])),
            "altitude": row["gps_alt_m"], "copyOf": row["duplicate_of"],
            "text": text[:TEXT_PREVIEW] + (" ..." if len(text) > TEXT_PREVIEW else ""),
        }
        lat, lon = float(row["gps_lat"]), float(row["gps_lon"])
        place = places.setdefault((lat, lon), {
            "lat": lat, "lon": lon, "photos": [],
            "link": f"https://www.google.com/maps?q={lat},{lon}"})
        place["photos"].append(photo)

    with open(os.path.join(APP_DIR, "vendor", "leaflet.js"), encoding="utf-8") as handle:
        leaflet_js = handle.read()
    with open(os.path.join(APP_DIR, "vendor", "leaflet.css"), encoding="utf-8") as handle:
        leaflet_css = handle.read()
    # "</" must not appear inside a <script> block; "<\/" means the same in JavaScript.
    data = json.dumps(list(places.values()), ensure_ascii=False).replace("</", "<\\/")
    note = (f"{len(rows)} photos with a GPS position at {len(places)} places. "
            "Bigger circle = more photos at that exact position. "
            "Positions come from the photo metadata and can be edited or wrong.")
    page = (PAGE.replace("__TITLE__", html.escape(f"{bucket}: photo GPS positions"))
            .replace("__NOTE__", html.escape(note))
            .replace("__LEAFLET_CSS__", leaflet_css)
            .replace("__LEAFLET_JS__", leaflet_js.replace("</script", "<\\/script"))
            .replace("__DATA__", data))
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(page)
    return out_path, len(rows), len(places)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for csv_path in sys.argv[1:]:
        result = build_map(csv_path)
        if result is None:
            print(f"{csv_path}: no rows with a GPS position")
        else:
            print(f"{csv_path}: {result[1]} photos at {result[2]} places. Map: {result[0]}")


if __name__ == "__main__":
    main()
