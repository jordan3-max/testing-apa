import os
import time
import csv
import argparse
import logging
import random
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
UA = "UniversityLogoScraper/1.0 (https://example.tld) contact@example.tld"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def make_session(retries=5, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504)):
    s = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=("GET", "POST"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": UA})
    return s


class RateLimiter:
    """Simple rate limiter (min interval between requests) with jitter support."""

    def __init__(self, min_interval=0.15):
        self.min_interval = float(min_interval)
        self.lock = threading.Lock()
        self._last = 0.0

    def wait(self):
        with self.lock:
            now = time.time()
            wait_for = self.min_interval - (now - self._last)
            if wait_for > 0:
                # add a small random jitter to avoid bursts
                jitter = random.uniform(0, min(0.1, self.min_interval * 0.2))
                time.sleep(wait_for + jitter)
            self._last = time.time()

def sparql_page(limit=500, offset=0, session=None, rate_limiter=None, timeout=60):
    """Query Wikidata SPARQL service and return bindings list.

    Uses provided `session` (requests.Session) and `rate_limiter` (RateLimiter) if given.
    Respects Retry-After header on 429.
    """
    q = """
    SELECT ?item ?itemLabel ?logo WHERE {
      ?item wdt:P31/wdt:P279* wd:Q3918 .
      ?item wdt:P154 ?logo .
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    LIMIT %d OFFSET %d
    """ % (limit, offset)
    headers = {"User-Agent": UA, "Accept": "application/sparql-results+json"}
    s = session or requests
    if rate_limiter:
        rate_limiter.wait()
    try:
        r = s.get(WIKIDATA_SPARQL, params={"query": q}, headers=headers, timeout=timeout)
    except Exception:
        raise
    if r.status_code == 429:
        ra = r.headers.get("Retry-After")
        wait = float(ra) if ra and ra.isdigit() else 5.0
        logging.warning("Received 429 from WDQS, sleeping %s seconds", wait)
        time.sleep(wait + random.uniform(0, 2.0))
        r.raise_for_status()
    r.raise_for_status()
    return r.json().get("results", {}).get("bindings", [])

def commons_imageinfo_from_title(title, session=None, rate_limiter=None, timeout=60):
    # title should be like "File:Name.ext"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "imageinfo",
        "iiprop": "url|mime|extmetadata",
    }
    headers = {"User-Agent": UA}
    s = session or requests
    if rate_limiter:
        rate_limiter.wait()
    r = s.get(COMMONS_API, params=params, headers=headers, timeout=timeout)
    if r.status_code == 429:
        ra = r.headers.get("Retry-After")
        wait = float(ra) if ra and ra.isdigit() else 5.0
        logging.warning("Received 429 from Commons API, sleeping %s seconds", wait)
        time.sleep(wait + random.uniform(0, 2.0))
        r = s.get(COMMONS_API, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    pages = data.get("query", {}).get("pages", {})
    for p in pages.values():
        if "imageinfo" in p:
            return p["imageinfo"][0]
    return None

def resolve_logo_value(val, session=None, rate_limiter=None):
    # SPARQL may return an http URL (Special:FilePath) or a File: title
    v = val
    if v.startswith("http"):
        return v  # likely Special:FilePath direct URL
    if v.startswith("File:") or v.startswith("Image:"):
        info = commons_imageinfo_from_title(v, session=session, rate_limiter=rate_limiter)
        if info:
            return info.get("url"), info.get("extmetadata", {})
    # fallback: try to form Special:FilePath
    if v.startswith("http://commons.wikimedia.org/wiki/File:") or v.startswith("https://commons.wikimedia.org/wiki/File:"):
        # convert to Special:FilePath link
        filename = v.split("/wiki/")[-1]
        return f"https://commons.wikimedia.org/wiki/Special:FilePath/{filename}"
    return None

def download_url(url, outpath, session=None, headers=None, timeout=60):
    headers = headers or {"User-Agent": UA}
    s = session or requests
    try:
        with s.get(url, stream=True, headers=headers, timeout=timeout) as r:
            r.raise_for_status()
            os.makedirs(os.path.dirname(outpath), exist_ok=True)
            with open(outpath, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return True, None
    except Exception as e:
        return False, str(e)

def worker(item_label, logo_val, outdir, master_session, rate_limiter, idx, total, per_thread_retries=3):
    # logo_val is the raw SPARQL value string
    try:
        # create a per-thread session to avoid sharing connection pools across threads too aggressively
        thread_session = make_session(retries=per_thread_retries, backoff_factor=0.5)
        # resolve to direct url (may return tuple (url, extmetadata) or url string)
        resolved = resolve_logo_value(logo_val, session=master_session, rate_limiter=rate_limiter)
        url = None
        extmeta = {}
        if isinstance(resolved, tuple):
            url, extmeta = resolved
        elif isinstance(resolved, str):
            url = resolved
        if not url:
            return (item_label, None, "no-url")
        # safe filename: index + original basename
        basename = os.path.basename(url.split("?")[0])
        safe_name = f"{idx:06d}_{sanitize_filename(item_label)}_{basename}"
        outpath = os.path.join(outdir, safe_name)
        if os.path.exists(outpath):
            return (item_label, outpath, "exists")
        ok, err = download_url(url, outpath, session=thread_session)
        # small polite delay between downloads
        if rate_limiter:
            rate_limiter.wait()
        else:
            time.sleep(0.15)
        if ok:
            lic = extract_license(extmeta)
            return (item_label, outpath, lic or "ok")
        else:
            return (item_label, None, f"download-error:{err}")
    except Exception as e:
        return (item_label, None, f"error:{str(e)}")

def sanitize_filename(name):
    # simple sanitizer
    return "".join(c if c.isalnum() or c in " -_." else "_" for c in name).strip()

def extract_license(extmeta):
    if not extmeta:
        return None
    lic = extmeta.get("LicenseShortName") or extmeta.get("License")
    if isinstance(lic, dict) and "value" in lic:
        return lic["value"]
    return lic

def main(args):
    outdir = args.out
    os.makedirs(outdir, exist_ok=True)
    csvpath = os.path.join(outdir, "metadata.csv")
    # CSV header
    with open(csvpath, "a", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        if cf.tell() == 0:
            writer.writerow(["university", "file_path", "status_or_license"])
    offset = 0
    limit = args.limit
    idx = 0
    pbar = None
    # master session for SPARQL/Commons API requests
    master_session = make_session(retries=args.retries, backoff_factor=0.5)
    master_session.headers.update({"User-Agent": UA})
    # rate limiter controls min interval between outgoing requests across threads
    rate_limiter = RateLimiter(min_interval=args.min_interval)

    # build set of already-downloaded universities (resume support)
    existing = set()
    if args.resume and os.path.exists(csvpath):
        try:
            with open(csvpath, newline="", encoding="utf-8") as cf:
                reader = csv.reader(cf)
                next(reader, None)
                for r in reader:
                    if r:
                        existing.add(r[0])
        except Exception:
            logging.exception("Failed reading existing metadata.csv for resume; continuing without resume")

    try:
        while True:
            logging.info("Requesting SPARQL page: limit=%s offset=%s", limit, offset)
            bindings = sparql_page(limit=limit, offset=offset, session=master_session, rate_limiter=rate_limiter)
            if not bindings:
                logging.info("No more bindings returned; exiting loop")
                break
            if pbar is None:
                pbar = tqdm(total=0, unit="files", desc="downloaded", leave=True)
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = []
                for b in bindings:
                    idx += 1
                    label = b.get("itemLabel", {}).get("value", b.get("item", {}).get("value", "unknown"))
                    if label in existing:
                        logging.debug("Skipping existing: %s", label)
                        continue
                    logo_val = b.get("logo", {}).get("value", "")
                    futures.append(ex.submit(worker, label, logo_val, outdir, master_session, rate_limiter, idx, None, args.retries))
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"batch {offset}", unit="item"):
                    res_label, path, status = fut.result()
                    with open(csvpath, "a", newline="", encoding="utf-8") as cf:
                        writer = csv.writer(cf)
                        writer.writerow([res_label, path or "", status])
                    if status not in ("exists", "no-url"):
                        pbar.update(1)
            # advance offset by actual returned count (not the requested limit)
            offset += len(bindings)
            # be polite between SPARQL pages
            time.sleep(1.0)
    finally:
        if pbar:
            pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape university logos from Wikidata / Commons (be polite, check licenses).")
    parser.add_argument("--out", "-o", default="logos", help="output directory")
    parser.add_argument("--workers", "-w", type=int, default=4, help="parallel downloads")
    parser.add_argument("--limit", "-l", type=int, default=500, help="SPARQL page size (use a conservative value like 500)")
    parser.add_argument("--min-interval", "-i", type=float, default=0.15, help="min interval (s) between requests across threads")
    parser.add_argument("--retries", "-r", type=int, default=3, help="per-request retries/backoff")
    parser.add_argument("--resume", action="store_true", help="resume using existing metadata.csv to skip already-downloaded entries")
    args = parser.parse_args()
    main(args)