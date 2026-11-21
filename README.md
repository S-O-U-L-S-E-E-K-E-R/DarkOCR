# DarkOCR

Lists a public S3-style storage bucket, downloads every image, reads the text in each image
on the GPU, extracts the metadata stored inside the photo (capture time, camera, GPS
position), and writes everything to a CSV, a short summary and an interactive map.

Works with Amazon S3 and the services that copy its interface: Aliyun OSS, Tencent COS,
Google Cloud Storage, DigitalOcean Spaces, MinIO and others. All requests are plain
anonymous HTTPS GETs; no cloud account or SDK is involved.

Use it only on data you are authorised to collect. The output contains whatever personal
data the images contain. Store and share it accordingly.

## Start

```bash
git clone https://github.com/S-O-U-L-S-E-E-K-E-R/DarkOCR.git
cd DarkOCR
./run.sh
```

To update an existing copy: `git pull`, then `./run.sh`. New packages are installed
automatically, and GPU engines are rebuilt when a new version needs them (about 10 minutes,
once). `./run.sh --tune` afterwards makes text detection about 30% faster.

The first start on a machine runs `setup.sh`, which creates `.venv` and installs the right
packages for that machine. Then it asks four things; press Enter to accept a default.

```
Bucket name(s) or address(es): mybucket.oss-cn-hongkong.aliyuncs.com
Keys file(s), blank to list the whole bucket:
Save CSV to [results/mybucket.csv]:
Keep images in [images/mybucket]:
```

A bucket can be given as an S3 bucket name, a bucket hostname, or any URL copied from the
bucket. Several buckets, separated by spaces, are processed one after another.

Stop with Ctrl+C at any time. Running again continues where it stopped.

## What it writes

| Path | Content |
|---|---|
| `keys/<bucket>.tsv` | Every key in the bucket: modified time, size, etag, key |
| `results/<bucket>.csv` | One row per image (columns below) |
| `results/<bucket>.summary.txt` | Counts, capture date range, cameras, list of GPS positions |
| `results/<bucket>.map.html` | Map of all photos with a GPS position. Open in a browser |
| `results/<bucket>.csv.errors.txt` | Images that failed. They are retried on the next run |
| `images/<bucket>/` | The downloaded images, in the bucket's folder structure |
| `engines/` | GPU engines built for this machine on first use |

CSV columns: `image_id`, `text`, `sha256`, `size_bytes`, `last_modified`, `etag`,
`duplicate_of`, `width`, `height`, `format`, `taken_at`, `gps_lat`, `gps_lon`, `gps_alt_m`,
`gps_time_utc`, `map_url`, `camera_make`, `camera_model`, `software`, `metadata_json`.
`metadata_json` holds every tag found in the file (EXIF, GPS, XMP, comments).

Files with the same etag are byte-identical. Only the first is downloaded and read; the
others get the same result with `duplicate_of` filled in. `--no-dedupe` turns this off.

## Requirements

- Linux, Python 3.10 to 3.13 with the `venv` module.
- For GPU speed: an NVIDIA GPU with at least 2 GB of memory and **driver 550 or newer**.
  No system-wide CUDA install is needed; the CUDA libraries are installed into `.venv`.
- About 8 GB of disk for the GPU setup, 1 GB for the CPU setup.

`setup.sh` picks the setup from what `nvidia-smi` reports:

| Machine | Setup | Text engine |
|---|---|---|
| NVIDIA GPU, driver 550+ | GPU | TensorRT engines, built for that GPU on first start (about 10 minutes, once) |
| NVIDIA GPU, older driver | CPU, with a note to update the driver | ONNX Runtime on the CPU |
| No NVIDIA GPU | CPU | ONNX Runtime on the CPU |

At every start the tool tries the engines fastest first (TensorRT, then ONNX Runtime with
CUDA, then CPU) and uses the first one that starts, so a GPU that TensorRT cannot serve
still works, only slower. The number of parallel text-reading processes is chosen from
the free GPU memory.

Tested on one machine only: RTX 4050 Laptop (6 GB), driver 550. Other NVIDIA GPUs use the
same packages but are untested.

### Speed

Measured on that machine, reading from disk, three processes sharing the GPU:

| Images | Per second |
|---|---|
| ICDAR 2015 street photos enlarged to 3840x2160 (few text lines) | about 45 |
| FUNSD scanned forms (about 50 text lines each) | about 21 |

Each process decodes images in four threads, ahead of the one thread that reads them: the
decoding runs on the CPU while the reading waits for the GPU, so the GPU is never idle
waiting for an image.

Downloading is often the slower part: at 25 MB/s a line delivers about 12 photos of 2 MB
per second.

The GPU engine (`app/gpu_engine.py`) runs the stock PP-OCRv6 models with the work around
them moved onto the GPU: pixel normalisation is part of the detection engine, the
recognition engine returns the chosen characters instead of the full score table, and the
orientation model runs in full precision. That is about twice as fast as running the same
models through RapidOCR's own TensorRT support, with the same text: on 300 real photos both
found 9,510 words and agreed on 98.7% of them (the rest is rounding noise in both directions).

`./run.sh --tune` builds a second detection engine with TensorRT's slowest, most thorough
kernel search. It takes 10 to 30 minutes with an idle GPU, is done once, and is used
automatically afterwards. It made detection about 30% faster here.

The GPU engine covers the default text model (`ch`: Chinese + English). Other `--lang`
models run on the ONNX Runtime engine, which is slower.

Force a setup with `./setup.sh --gpu` or `./setup.sh --cpu`. Test an install with
`./run.sh --check`.

## Options

`./run.sh --help` lists all of them. The most useful:

| Option | Meaning |
|---|---|
| `--bucket A B ...` | Buckets, without the questions |
| `--keys FILE` | Read only the keys in this file (one key per line, or a `keys/*.tsv` file) |
| `--list-only` | Write the keys file and the bucket overview, download nothing |
| `--no-keep` | Delete each image after reading it |
| `--proxy URL` | Send all requests through a proxy, for example `socks5h://127.0.0.1:9050` |
| `--lang NAME` | Text model: `ch` (Chinese + English, default), `en`, `japan`, `korean`, `latin`, `cyrillic`, `arabic`, ... |
| `--max-side N` | Shrink bigger images to N pixels before reading (default 2000, 0 = full size) |
| `--engine NAME` | `auto` (default), `tensorrt`, `onnx`, `cpu` |
| `--tune` | Build the faster detection engine for this GPU, then stop |
| `--workers N` | Parallel downloads (default 32) |

`./map.sh results/<bucket>.csv` rebuilds a map from an existing CSV.

## Limits

- The bucket must allow anonymous reads. If listing is denied but single files are
  readable, supply the keys in a file.
- Text accuracy is good on clear print and screenshots, weaker on small, blurred or
  stylised text. Treat the CSV as a search index and check the image before relying on it.
- GPS positions and capture times come from the photo metadata, which can be missing,
  stripped, edited or wrong.
- The map file loads its background map from Esri (or OpenStreetMap when selected), so
  that provider sees which areas are viewed. The photos and positions stay in the file.

## Layout

```
run.sh          start (installs on first start)
setup.sh        install or reinstall .venv for this machine
map.sh          rebuild a GPS map from a results CSV
app/            the program: s3_ocr.py, gpu_engine.py, gps_map.py, vendor/ (Leaflet 1.9.4)
requirements/   package lists: base, gpu-cuda12, cpu
```

The text models are PP-OCR, run through [RapidOCR](https://github.com/RapidAI/RapidOCR);
the model files are downloaded by RapidOCR on first use.
