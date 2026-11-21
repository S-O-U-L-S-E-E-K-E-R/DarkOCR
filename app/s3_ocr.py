#!/usr/bin/env python3
"""List public S3-style buckets, download their images, read the text in each, write a CSV.

Works with Amazon S3 and the storage services that copy its interface: Aliyun OSS,
Tencent COS, Google Cloud Storage, DigitalOcean Spaces, MinIO and others.

Run with no arguments to be asked for the bucket(s). Press Enter at the other prompts to
accept the defaults, which are all inside this script's own folder:
    keys/<bucket>.tsv           every key in the bucket (modified, size, etag, key)
    results/<bucket>.csv        one row per image: text, hash, and the metadata inside the photo
    results/<bucket>.summary.txt  counts, date range, cameras, photos with GPS position
    images/<bucket>/            the downloaded images
    engines/                    GPU engines built on first use (takes several minutes, once)

All requests are plain anonymous HTTPS GETs with a browser User-Agent: no account,
no request signing, no vendor SDK. Use --proxy to hide the source IP address as well.
"""

import argparse
import collections
import csv
import hashlib
import fcntl
import json
import multiprocessing
import os
import re
import readline  # noqa: F401  (gives input() line editing)
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ElementTree
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import requests

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
CSV_COLUMNS = ["image_id", "text", "sha256", "size_bytes", "last_modified", "etag", "duplicate_of",
               "width", "height", "format", "taken_at", "gps_lat", "gps_lon", "gps_alt_m",
               "gps_time_utc", "map_url", "camera_make", "camera_model", "software", "metadata_json"]
MAX_METADATA_VALUE = 20000  # characters kept per metadata value
csv.field_size_limit(1 << 30)
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
REQUEST_ATTEMPTS = 4
# GPU memory one text-reading process needs, in MB (measured on an RTX 4050, 6 GB).
GPU_MB_PER_PROCESS = {"tensorrt": 1000, "onnx": 4000}
MAX_GPU_PROCESSES = 3  # more than 3 gave no extra speed: the processes take turns on the GPU
CPU_PROCESSES = 4
APP_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_DIR = os.path.dirname(APP_DIR)  # the project folder: keys/, results/, images/, engines/
ENGINE_DIR = os.path.join(HOME_DIR, "engines")


class BucketError(Exception):
    """A problem with one bucket that the user can act on."""


def make_session(proxy, connections=32):
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    adapter = requests.adapters.HTTPAdapter(pool_maxsize=connections)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    if proxy:
        session.proxies = {"http": proxy, "https": proxy}
    return session


def get_with_retries(session, url, **options):
    """GET that retries network errors and server-side errors, but not 403 or 404."""
    for attempt in range(1, REQUEST_ATTEMPTS + 1):
        try:
            response = session.get(url, timeout=60, **options)
            if response.status_code < 500 and response.status_code != 429:
                return response
            response.close()
            problem = f"HTTP {response.status_code}"
        except requests.RequestException as error:
            problem = str(error)
        if attempt == REQUEST_ATTEMPTS:
            raise BucketError(f"{url}: {problem}")
        time.sleep(2 ** attempt)


def parse_listing(content):
    """Return the XML root of a bucket listing with namespaces removed, or None."""
    try:
        root = ElementTree.fromstring(content)
    except ElementTree.ParseError:
        return None
    for element in root.iter():  # each vendor uses its own XML namespace, so drop it
        element.tag = element.tag.rsplit("}", 1)[-1]
    return root if root.tag == "ListBucketResult" else None


def find_region(session, bucket):
    """S3 reports the bucket region in a response header, even on redirects and denials."""
    response = session.head(f"https://s3.amazonaws.com/{bucket}", timeout=30, allow_redirects=False)
    return response.headers.get("x-amz-bucket-region") or "us-east-1"


def resolve_bucket(session, value):
    """Turn what the user typed into (name for output files, base URL).

    Amazon S3: bare bucket name, s3://name, or any S3 hostname or URL.
    Any other storage: the bucket hostname or any URL from the bucket. Both address forms
    work: bucket in the hostname (name.oss-cn-hongkong.aliyuncs.com) and bucket as the
    first part of the path (storage.googleapis.com/name).
    """
    value = value.strip().strip("'\"")
    scheme = "http" if value.startswith("http://") else "https"
    host, _, path = re.sub(r"^(s3|https?)://", "", value).partition("/")
    first_part = path.split("/")[0].split("?")[0]

    match = re.match(r"^(.+?)\.s3[.-]([a-z0-9-]+)\.amazonaws\.com$", host)
    path_style_s3 = re.match(r"^s3[.-]([a-z0-9-]+\.)?amazonaws\.com$", host)
    if path_style_s3 and first_part:
        host, match = first_part, None
    if match or "." not in host or host.endswith(".s3.amazonaws.com"):
        bucket = match.group(1) if match else host.removesuffix(".s3.amazonaws.com")
        region = match.group(2) if match else find_region(session, bucket)
        if "." in bucket:  # dots break the TLS certificate match for the bucket-as-hostname form
            return bucket, f"https://s3.{region}.amazonaws.com/{bucket}"
        return bucket, f"https://{bucket}.s3.{region}.amazonaws.com"

    host_form = (host.split(".")[0].split(":")[0], f"{scheme}://{host}")
    if not first_part:
        return host_form
    # A URL with a path: the bucket is either the hostname or the first part of the path.
    # Ask the server which of the two answers with a bucket listing.
    path_form = (first_part, f"{scheme}://{host}/{first_part}")
    for name, base_url in (host_form, path_form):
        try:
            response = get_with_retries(session, base_url + "/", params={"max-keys": "1"})
        except BucketError:
            continue
        if response.ok and parse_listing(response.content) is not None:
            return name, base_url
    return host_form


def list_bucket(session, base_url):
    """Yield (key, modified, size, etag) for every object. No limit on bucket size.

    Uses the marker-style listing, which Amazon S3 and all the copies of it answer.
    """
    params = {"max-keys": "1000"}
    while True:
        response = get_with_retries(session, base_url + "/", params=params)
        if response.status_code in (401, 403):
            raise BucketError(f"{base_url}: listing denied (HTTP {response.status_code}). "
                              "If single files are still readable, supply a keys file.")
        if response.status_code == 404:
            raise BucketError(f"{base_url}: no such bucket (HTTP 404).")
        root = parse_listing(response.content)
        if not response.ok or root is None:
            raise BucketError(f"{base_url}: not a bucket listing (HTTP {response.status_code}).")
        last_key = None
        for item in root.iter("Contents"):
            last_key = item.findtext("Key")
            yield (last_key, item.findtext("LastModified") or "", int(item.findtext("Size") or 0),
                   (item.findtext("ETag") or "").strip('"'))
        next_marker = root.findtext("NextMarker") or last_key
        if root.findtext("IsTruncated") != "true" or not next_marker \
                or next_marker == params.get("marker"):
            return
        params["marker"] = next_marker


def list_and_save(session, bucket, base_url, keys_path):
    """List the whole bucket into a keys file, print a summary, return {key: (modified, etag)}."""
    os.makedirs(os.path.dirname(os.path.abspath(keys_path)), exist_ok=True)
    keys = {}
    total_bytes = 0
    prefix_count, prefix_bytes, types = collections.Counter(), collections.Counter(), collections.Counter()
    try:
        with open(keys_path + ".part", "w", encoding="utf-8") as handle:
            for key, modified, size, etag in list_bucket(session, base_url):
                handle.write(f"{modified}\t{size}\t{etag}\t{key}\n")
                keys[key] = (modified, etag)
                total_bytes += size
                prefix = key.split("/")[0] if "/" in key else "(top level)"
                prefix_count[prefix] += 1
                prefix_bytes[prefix] += size
                name = key.rsplit("/", 1)[-1]
                types[name.rsplit(".", 1)[-1].lower() if "." in name else "(none)"] += 1
                if len(keys) % 10000 == 0:
                    print(f"  listed {len(keys)} keys ...", flush=True)
    except BaseException:
        os.remove(keys_path + ".part")  # do not leave a half-written listing behind
        raise
    os.replace(keys_path + ".part", keys_path)

    print(f"Bucket {bucket}: {len(keys)} keys, {total_bytes / 1e6:.1f} MB. Keys file: {keys_path}")
    for prefix, count in prefix_count.most_common(15):
        print(f"  {count:>8}  {prefix_bytes[prefix] / 1e6:>10.1f} MB  {prefix}")
    print("  File types: " + ", ".join(f"{name} {count}" for name, count in types.most_common(10)))
    return keys


def keys_from_files(paths):
    """Return {key: (modified, etag)}. One key per line; with tab-separated columns the key is
    the last column, and a keys file written by this script also gives modified and etag."""
    keys = {}
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                columns = line.rstrip("\r\n").split("\t")
                if columns[-1]:
                    modified = columns[0] if len(columns) >= 3 else ""
                    etag = columns[2] if len(columns) >= 4 else ""
                    keys.setdefault(columns[-1], (modified, etag))
    return keys


def rows_already_done(csv_path):
    """Return {image_id: row} from an earlier run. A CSV with other columns is set aside."""
    if not os.path.exists(csv_path):
        return {}
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames == CSV_COLUMNS:
            return {row["image_id"]: row for row in reader}
    old_path = f"{csv_path}.old-{time.strftime('%Y%m%d-%H%M%S')}"
    os.replace(csv_path, old_path)
    print(f"Existing CSV has different columns; moved to {old_path}")
    return {}


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(session, base_url, key, local_path):
    """Download one key unless a kept copy exists. Return the local path."""
    if os.path.exists(local_path):
        return local_path
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with get_with_retries(session, base_url + "/" + urllib.parse.quote(key), stream=True) as response:
        response.raise_for_status()
        with open(local_path + ".part", "wb") as handle:
            for chunk in response.iter_content(1 << 16):
                handle.write(chunk)
    os.replace(local_path + ".part", local_path)
    return local_path


def plain(value):
    """Make one metadata value safe for JSON."""
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8").strip("\x00 ")
        except UnicodeDecodeError:
            return f"({len(value)} bytes) " + value[:32].hex()
    if isinstance(value, str):
        return value.replace("\x00", "")[:MAX_METADATA_VALUE]
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (int, bool)) or value is None:
        return value
    try:
        number = float(value)  # EXIF rationals
        return number if number == number and abs(number) != float("inf") else str(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return str(value)[:MAX_METADATA_VALUE]


def degrees(parts, direction):
    """EXIF (degrees, minutes, seconds) plus N/S/E/W to signed decimal degrees."""
    d, m, sec = (float(part) for part in parts)
    value = d + m / 60 + sec / 3600
    return round(-value if str(direction).strip().upper() in ("S", "W") else value, 7)


def read_metadata(path):
    """Return the CSV metadata columns for one image file. Never raises."""
    from PIL import ExifTags, Image
    Image.MAX_IMAGE_PIXELS = None  # only headers are read here, so huge photos are fine
    row = {}
    everything = {}
    try:
        with Image.open(path) as image:
            row.update(width=image.width, height=image.height, format=image.format)
            exif = image.getexif()
            base = {ExifTags.TAGS.get(tag, str(tag)): value for tag, value in exif.items()}
            detail = {ExifTags.TAGS.get(tag, str(tag)): value
                      for tag, value in exif.get_ifd(ExifTags.IFD.Exif).items()}
            gps = {ExifTags.GPSTAGS.get(tag, str(tag)): value
                   for tag, value in exif.get_ifd(ExifTags.IFD.GPSInfo).items()}
            for name, group in (("exif", base), ("exif_detail", detail), ("gps", gps)):
                if group:
                    everything[name] = plain(group)
            for name, value in image.info.items():  # XMP, comments, PNG text, DPI, ...
                if name not in ("exif", "icc_profile"):
                    everything.setdefault("other", {})[str(name)] = plain(value)
            if image.info.get("icc_profile"):
                everything.setdefault("other", {})["icc_profile_bytes"] = len(image.info["icc_profile"])
    except Exception as error:
        everything["metadata_error"] = str(error)
        base = detail = gps = {}

    def text(value):
        return str(plain(value)).strip() if value not in (None, "") else ""

    row["taken_at"] = text(detail.get("DateTimeOriginal") or detail.get("DateTimeDigitized")
                           or base.get("DateTime"))
    row["camera_make"] = text(base.get("Make"))
    row["camera_model"] = text(base.get("Model"))
    row["software"] = text(base.get("Software"))
    try:
        if gps.get("GPSLatitude") and gps.get("GPSLongitude"):
            lat = degrees(gps["GPSLatitude"], gps.get("GPSLatitudeRef", "N"))
            lon = degrees(gps["GPSLongitude"], gps.get("GPSLongitudeRef", "E"))
            if (lat, lon) != (0, 0) and abs(lat) <= 90 and abs(lon) <= 180:
                row.update(gps_lat=lat, gps_lon=lon,
                           map_url=f"https://www.google.com/maps?q={lat},{lon}")
        if gps.get("GPSAltitude") is not None:
            altitude = float(gps["GPSAltitude"])
            below_sea = gps.get("GPSAltitudeRef") in (1, b"\x01")
            row["gps_alt_m"] = round(-altitude if below_sea else altitude, 1)
        if gps.get("GPSDateStamp") and gps.get("GPSTimeStamp"):
            h, m, sec = (float(part) for part in gps["GPSTimeStamp"])
            row["gps_time_utc"] = (f"{str(gps['GPSDateStamp']).strip().replace(':', '-')}"
                                   f"T{int(h):02}:{int(m):02}:{int(sec):02}Z")
    except (TypeError, ValueError, ZeroDivisionError):
        pass  # malformed GPS values stay available in metadata_json
    row["metadata_json"] = json.dumps(everything, ensure_ascii=False, sort_keys=True) if everything else ""
    return row


def load_image(path, max_side):
    """Decode an image, shrunk so its longest side is at most max_side (0 = full size)."""
    import cv2
    import numpy
    flag = cv2.IMREAD_COLOR
    if max_side and max_side <= 2000 and os.path.getsize(path) > 1_500_000:
        flag = cv2.IMREAD_REDUCED_COLOR_2  # big photo: decode at half size, much faster
    image = cv2.imread(path, flag)
    if image is None:  # formats OpenCV cannot read
        from PIL import Image
        with Image.open(path) as picture:
            image = cv2.cvtColor(numpy.asarray(picture.convert("RGB")), cv2.COLOR_RGB2BGR)
    if max_side:
        scale = max_side / max(image.shape[:2])
        if scale < 1:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image


class TextReader:
    """PP-OCR text reading. One instance per process, used by one thread.

    Engines, fastest first:
      tensorrt  gpu_engine.py: TensorRT engines with pre- and post-processing on the GPU.
      onnx      RapidOCR on ONNX Runtime with CUDA.
      cpu       RapidOCR on ONNX Runtime without a GPU.
    "auto" tries them in that order and uses the first that starts.
    """

    def __init__(self, language, engine):
        import cv2
        import numpy
        cv2.setNumThreads(1)  # several processes run side by side; do not oversubscribe cores
        blank = numpy.full((128, 256, 3), 255, numpy.uint8)
        cv2.putText(blank, "test 123", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        self.engine_name = None
        self.language = language
        self.fallback = None
        self.tuned = False
        problems = []
        for name in (["tensorrt", "onnx", "cpu"] if engine == "auto" else [engine]):
            try:
                self.engine = self.build(name, language)
                self.engine(blank)  # loads all three models, proves the engine really runs
                self.engine_name = name
                self.tuned = getattr(self.engine, "tuned", False)
                break
            except Exception as error:
                problems.append(f"{name}: {str(error)[:200]}")
        if self.engine_name is None:
            raise RuntimeError("no text engine could start. " + " | ".join(problems))
        self.problems = problems

    @staticmethod
    def build(name, language):
        if name == "tensorrt":
            import gpu_engine  # next to this file
            return gpu_engine.GpuOCR(language)
        import onnxruntime
        from rapidocr import RapidOCR
        params = {"Global.log_level": "critical", "Rec.lang_type": language,
                  "Rec.rec_batch_num": 16, "Cls.cls_batch_num": 16}
        if name == "cpu":
            params["EngineConfig.onnxruntime.intra_op_num_threads"] = 4
        if name == "onnx":
            onnxruntime.preload_dlls()  # use the CUDA libraries installed in this environment
            cuda = "EngineConfig.onnxruntime.cuda_ep_cfg."
            params.update({"EngineConfig.onnxruntime.use_cuda": True,
                           cuda + "cudnn_conv_algo_search": "HEURISTIC",
                           cuda + "arena_extend_strategy": "kSameAsRequested"})
            if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
                raise RuntimeError("ONNX Runtime has no CUDA support")
        return RapidOCR(params=params)

    def read(self, image):
        try:
            result = self.engine(image)
        except Exception:
            # GPU memory under load, or an input the GPU engine cannot take (for example a
            # text line wider than the recognition engine was built for): read it on the CPU.
            if self.fallback is None:
                self.fallback = self.engine if self.engine_name == "cpu" else self.build("cpu", self.language)
            result = self.fallback(image)
        return " ".join(result.txts) if result.txts else ""


_reader = None  # the TextReader of this worker process


def start_worker(language, engine):
    """Runs once in every text-reading process."""
    global _reader
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C is handled by the main process
    os.makedirs(ENGINE_DIR, exist_ok=True)
    with open(os.path.join(ENGINE_DIR, ".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # one process builds the engines, the others reuse them
        _reader = TextReader(language, engine)


def worker_engine():
    return _reader.engine_name + (" (tuned)" if _reader.tuned else ""), _reader.problems


def build_tuned_engine():
    import gpu_engine
    return gpu_engine.tune()


def analyze(local_path, max_side):
    """Metadata and text for one downloaded image. Runs in a text-reading process."""
    row = read_metadata(local_path)
    row["text"] = _reader.read(load_image(local_path, max_side))
    return row


def process(session, base_url, key, local_path, readers, max_side, keep):
    """Everything for one image: download, hash, then metadata and text. Runs in a thread."""
    fetch(session, base_url, key, local_path)
    row = {"image_id": key, "sha256": sha256_of(local_path), "size_bytes": os.path.getsize(local_path)}
    row.update(readers.submit(analyze, local_path, max_side).result())
    if not keep:
        os.remove(local_path)
    return row


def read_images(session, base_url, groups, total, out_path, keep_dir, readers, workers, max_side):
    """Process every group of identical files, appending rows to the CSV. Return the failure count.

    groups: list of [(key, modified, etag), ...]; the first key of a group is downloaded and
    read, the others are byte-identical copies and reuse its result.
    """
    errors_path = out_path + ".errors.txt"
    is_new_file = not os.path.exists(out_path)
    failures = 0
    with open(out_path, "a", newline="", encoding="utf-8") as handle, \
            tempfile.TemporaryDirectory() as temp_dir, \
            ThreadPoolExecutor(workers) as pool:
        writer = csv.DictWriter(handle, CSV_COLUMNS, restval="")
        if is_new_file:
            writer.writeheader()

        # Work runs ahead of the CSV writing, but only by a bounded number of images.
        image_dir = os.path.abspath(keep_dir or temp_dir)
        remaining = iter(groups)
        pending = collections.deque()

        def queue_next():
            group = next(remaining, None)
            if group is None:
                return
            key = group[0][0]
            local_path = os.path.normpath(os.path.join(image_dir, key))
            if not local_path.startswith(image_dir + os.sep):
                local_path = os.path.join(image_dir, hashlib.sha256(key.encode()).hexdigest())
            pending.append((group, pool.submit(process, session, base_url, key, local_path,
                                               readers, max_side, bool(keep_dir))))

        for _ in range(workers * 2):
            queue_next()
        number = 0
        while pending:
            group, work = pending.popleft()
            queue_next()
            first_key = group[0][0]
            try:
                row = work.result()
            except Exception as error:  # one bad image must not stop the run
                failures += len(group)
                number += len(group)
                with open(errors_path, "a", encoding="utf-8") as errors:
                    for key, _, _ in group:
                        errors.write(f"{key}\t{error}\n")
                print(f"[{number}/{total}] FAILED {first_key}: {error}", file=sys.stderr)
                continue
            for key, modified, etag in group:
                number += 1
                writer.writerow({**row, "image_id": key, "last_modified": modified, "etag": etag,
                                 "duplicate_of": "" if key == first_key else first_key})
            handle.flush()
            copies = f" (+{len(group) - 1} identical)" if len(group) > 1 else ""
            print(f"[{number}/{total}] {first_key}{copies}: {row['text'][:60]}")
    if failures:
        print(f"{failures} image(s) failed, listed in {errors_path}. Run again to retry them.",
              file=sys.stderr)
    return failures


def write_summary(csv_path, bucket):
    """Write a short text summary of the CSV next to it and print it."""
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    cameras = collections.Counter(" ".join(filter(None, [row["camera_make"], row["camera_model"]]))
                                  for row in rows if row["camera_make"] or row["camera_model"])
    software = collections.Counter(row["software"] for row in rows if row["software"])
    taken = sorted(row["taken_at"] for row in rows if row["taken_at"][:1].isdigit())
    located = [row for row in rows if row["gps_lat"]]
    lines = [
        f"Bucket: {bucket}",
        f"Images in CSV: {len(rows)}",
        f"Distinct files (by content): {len({row['sha256'] for row in rows})}",
        f"Images with text: {sum(bool(row['text']) for row in rows)}",
        f"Images with embedded metadata: {sum(bool(row['metadata_json']) for row in rows)}",
        f"Images with capture time: {len(taken)}" + (f" ({taken[0]} to {taken[-1]})" if taken else ""),
        f"Images with GPS position: {len(located)}",
        "Cameras: " + (", ".join(f"{name} ({count})" for name, count in cameras.most_common(15)) or "none"),
        "Software: " + (", ".join(f"{name} ({count})" for name, count in software.most_common(15)) or "none"),
    ]
    if located:
        lines.append("GPS positions (image_id, lat, lon, capture time):")
        lines += [f"  {row['image_id']}\t{row['gps_lat']}\t{row['gps_lon']}\t{row['taken_at']}"
                  for row in located]
    summary_path = os.path.splitext(csv_path)[0] + ".summary.txt"
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines[:9]))
    print(f"Summary: {summary_path}")
    if located:
        import gps_map  # next to this file
        print(f"Map: {gps_map.build_map(csv_path)[0]}")


def free_gpu_mb():
    """Free memory of the first NVIDIA GPU in MB, or None when it cannot be asked."""
    try:
        answer = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=20, check=True).stdout
        return int(answer.split()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def default_processes(engine):
    """How many text-reading processes fit this machine."""
    if engine == "cpu":
        return max(1, min(CPU_PROCESSES, (os.cpu_count() or 4) // 4))
    free = free_gpu_mb()
    if free is None:
        return 1
    return max(1, min(MAX_GPU_PROCESSES, (free - 300) // GPU_MB_PER_PROCESS[engine]))


def installed_mode():
    """"gpu" or "cpu": what setup.sh installed. Unknown counts as gpu."""
    try:
        with open(os.path.join(HOME_DIR, ".venv", "install-mode"), encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return "gpu"


def check_engine(language, engine):
    """Start the text engine, read a test image, report. Used by setup.sh and --check."""
    import cv2
    import numpy
    readers = start_readers(language, engine, 1)
    image = numpy.full((160, 640, 3), 255, numpy.uint8)
    cv2.putText(image, "Engine check 12345", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 3)
    path = os.path.join(tempfile.mkdtemp(), "check.png")
    cv2.imwrite(path, image)
    text = readers.submit(analyze, path, 2000).result()["text"]
    readers.shutdown()
    os.remove(path)
    good = "12345" in text
    print(f"Test image read as: {text!r}  ->  {'OK' if good else 'WRONG'}")
    return good


def worker_ready():
    time.sleep(1)  # long enough that every process of the pool takes one of these
    return _reader.engine_name


def start_readers(language, engine, processes):
    """Start the text-reading processes and report which engine they run."""
    if not os.path.exists(os.path.join(ENGINE_DIR, ".lock")) and engine in ("auto", "tensorrt"):
        print("First run: building GPU engines for this machine. This takes several minutes, once.")
    context = multiprocessing.get_context("spawn")
    # One process first: it builds missing engines and shows which engine really starts.
    with ProcessPoolExecutor(1, mp_context=context, initializer=start_worker,
                             initargs=(language, engine)) as probe:
        try:
            name, problems = probe.submit(worker_engine).result()
        except Exception as error:
            raise BucketError(f"text engine did not start: {error}") from error
    for problem in problems:
        print(f"Engine not usable, trying next. {problem}", file=sys.stderr)
    label, name = name, name.split()[0]
    if processes is None:
        processes = default_processes(name)
    readers = ProcessPoolExecutor(processes, mp_context=context, initializer=start_worker,
                                  initargs=(language, name))
    for ready in [readers.submit(worker_ready) for _ in range(processes)]:
        ready.result()
    print(f"Text engine: {label}, {processes} process(es)")
    return readers


def ask(prompt):
    return input(prompt).strip()


def ask_path(prompt, default):
    answer = split_paths(ask(f"{prompt} [{default}]: "))
    return answer[0] if answer else default


def split_paths(answer):
    """Paths separated by spaces; quoted paths (as pasted by drag and drop) may hold spaces."""
    return [os.path.expanduser(path) for path in shlex.split(answer)]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bucket", nargs="+",
                        help="one or more buckets: S3 bucket name, or bucket hostname/URL")
    parser.add_argument("--keys", nargs="*", help="keys file(s); omit to list the whole bucket")
    parser.add_argument("--keys-out", help="where to save the bucket listing")
    parser.add_argument("--list-only", action="store_true", help="save the keys file and stop")
    parser.add_argument("--out", help="output CSV path")
    parser.add_argument("--keep", help="folder to keep the downloaded images in")
    parser.add_argument("--no-keep", action="store_true", help="delete each image after reading it")
    parser.add_argument("--proxy", help="send all requests through this proxy, "
                                        "for example socks5h://127.0.0.1:9050 for Tor")
    parser.add_argument("--workers", type=int, default=32, help="parallel downloads")
    parser.add_argument("--ocr-processes", type=int,
                        help="text-reading processes (default: as many as fit the GPU memory, "
                             "at most 3; on the CPU at most 4)")
    parser.add_argument("--check", action="store_true",
                        help="start the text engine, read a test image, and stop")
    parser.add_argument("--tune", action="store_true",
                        help="build a faster detection engine for this GPU (10 to 30 minutes, once), "
                             "then stop; later runs use it automatically")
    parser.add_argument("--engine", choices=["auto", "tensorrt", "onnx", "cpu"], default="auto",
                        help="auto = fastest engine that starts (TensorRT, then ONNX+CUDA, then CPU)")
    parser.add_argument("--max-side", type=int, default=2000,
                        help="shrink bigger images to this many pixels before reading; 0 = full size")
    parser.add_argument("--lang", default="ch",
                        help="text model: ch (Chinese + English, default), en, japan, korean, "
                             "latin, cyrillic, arabic, chinese_cht, devanagari, th, el, ...")
    parser.add_argument("--no-dedupe", action="store_true",
                        help="download and read byte-identical files again instead of once")
    parser.add_argument("--limit", type=int, help="stop after this many images per bucket")
    parser.add_argument("--cpu", action="store_true", help="same as --engine cpu")
    args = parser.parse_args()
    if args.cpu or (args.engine == "auto" and installed_mode() == "cpu"):
        args.engine = "cpu"
    if args.tune:
        print("Building the tuned detection engine. This takes 10 to 30 minutes; the GPU should be idle.")
        with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
            print("Saved", pool.submit(build_tuned_engine).result())
        sys.exit(0)
    if args.check:
        sys.exit(0 if check_engine(args.lang, args.engine) else 1)

    interactive = not args.bucket
    buckets = args.bucket or re.split(r"[\s,]+", ask("Bucket name(s) or address(es): "))
    buckets = [bucket for bucket in buckets if bucket]
    if not buckets:
        sys.exit("No bucket given.")
    one_bucket = len(buckets) == 1
    if not one_bucket and (args.keys or args.keys_out or args.out or args.keep):
        sys.exit("--keys, --keys-out, --out and --keep work with one bucket only. "
                 "With several buckets each bucket gets its own default files.")

    if args.keys is not None:
        key_files = args.keys
    elif interactive and one_bucket and not args.list_only:
        key_files = split_paths(ask("Keys file(s), blank to list the whole bucket: "))
    else:
        key_files = []
    missing = [path for path in key_files if not os.path.isfile(path)]
    if missing:
        sys.exit(f"Keys file not found: {', '.join(missing)}")

    session = make_session(args.proxy, args.workers)
    readers = None
    started = time.monotonic()
    problems = []
    for bucket_answer in buckets:
        try:
            bucket, base_url = resolve_bucket(session, bucket_answer)
            print(f"\n=== {bucket}  ({base_url})")

            # Everything this script writes goes under its own folder unless told otherwise.
            ask_user = interactive and one_bucket and not args.list_only
            out_path = args.out or os.path.join(HOME_DIR, "results", bucket + ".csv")
            keep_dir = args.keep or os.path.join(HOME_DIR, "images", bucket)
            if ask_user and not args.out:
                out_path = ask_path("Save CSV to", out_path)
            if os.path.isdir(out_path):
                out_path = os.path.join(out_path, bucket + ".csv")
            if args.no_keep:
                keep_dir = None
            elif ask_user and not args.keep:
                keep_dir = ask_path("Keep images in", keep_dir)

            if key_files:
                all_keys = keys_from_files(key_files)
            else:
                keys_path = args.keys_out or os.path.join(HOME_DIR, "keys", bucket + ".tsv")
                all_keys = list_and_save(session, bucket, base_url, keys_path)
            if args.list_only:
                continue

            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            done = rows_already_done(out_path)
            todo = [(key, modified, etag) for key, (modified, etag) in all_keys.items()
                    if key.lower().endswith(IMAGE_EXTENSIONS) and key not in done]
            if args.limit:
                todo = todo[:args.limit]
            # Files with the same etag and size are byte-identical: read one, reuse the result.
            groups = {}
            for item in todo:
                identity = item[2] if item[2] and not args.no_dedupe else item[0]
                groups.setdefault(identity, []).append(item)
            groups = list(groups.values())
            print(f"{len(todo)} image(s) to read ({len(groups)} distinct), "
                  f"{len(done)} already in {out_path}")
            if not todo:
                continue
            if readers is None:  # slow to start, so only when there is something to read
                readers = start_readers(args.lang, args.engine, args.ocr_processes)
            bucket_started = time.monotonic()
            failures = read_images(session, base_url, groups, len(todo), out_path, keep_dir,
                                   readers, args.workers, args.max_side)
            if failures:
                problems.append(f"{bucket}: {failures} image(s) failed")
            seconds = time.monotonic() - bucket_started
            print(f"CSV: {out_path}  ({len(todo)} images in {seconds / 60:.1f} min, "
                  f"{1000 * seconds / len(todo):.0f} ms/image)")
            write_summary(out_path, bucket)
        except (BucketError, requests.RequestException, OSError) as error:
            problems.append(f"{bucket_answer}: {error}")  # one bad bucket must not stop the others
            print(f"SKIPPED {bucket_answer}: {error}", file=sys.stderr)

    if readers is not None:
        readers.shutdown()
    print(f"\nDone in {(time.monotonic() - started) / 60:.1f} min.")
    if problems:
        print("Problems:\n  " + "\n  ".join(problems), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Run again to continue where it stopped.", file=sys.stderr)
        for child in multiprocessing.active_children():
            child.kill()
        os._exit(130)  # do not wait for downloads still in flight
