#manually scraped news websites → added to raw/
"""
Targeted Nepali scrapers for confirmed working sites:
- nepalnews.com (25,426 URLs, confirmed Nepali content)
- gorkhapatra.org.np (small but working)
"""
import requests
import time
import re
import unicodedata
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from pathlib import Path

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "ne-NP,ne;q=0.9,en-US;q=0.8",
}


def fetch(url, timeout=20):
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        r.encoding = "utf-8"
        return r.text
    except Exception as e:
        print(f"    FAIL {url[:80]}: {e}")
        return None


def clean(text):
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def has_devanagari(text, threshold=0.10):
    if not text or len(text) < 20:
        return False
    deva = sum(1 for c in text if 'ऀ' <= c <= 'ॿ')
    return deva / len(text) >= threshold


def extract_text(html, selectors):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe"]):
        tag.decompose()
    for sel in selectors:
        found = soup.select_one(sel)
        if found:
            paras = found.find_all("p")
            if paras:
                text = "\n".join(p.get_text(" ", strip=True) for p in paras)
            else:
                text = found.get_text("\n", strip=True)
            text = clean(text)
            if has_devanagari(text) and len(text) > 100:
                return text
    # Fallback
    paras = soup.find_all("p")
    text = "\n".join(p.get_text(" ", strip=True) for p in paras if len(p.get_text()) > 30)
    text = clean(text)
    return text if has_devanagari(text) and len(text) > 100 else None


# ── Nepal News (WordPress site) ──────────────────────────────────────────────

def get_nepalnews_urls(max_files=13):
    """Fetch all WordPress sitemap files for nepalnews.com."""
    all_urls = []
    for i in range(1, max_files + 1):
        r = fetch(f"https://nepalnews.com/wp-sitemap-posts-post-{i}.xml")
        if not r:
            break
        locs = re.findall(r"<loc>(.*?)</loc>", r)
        # Keep only Nepali-titled URLs (URL-encoded Devanagari)
        all_urls.extend(locs)
        print(f"    Sitemap file {i}: {len(locs)} URLs (total: {len(all_urls)})")
        time.sleep(0.3)
    return all_urls


def scrape_nepalnews(url):
    html = fetch(url)
    if not html:
        return None
    return extract_text(html, [
        ".entry-content",
        ".post-content",
        ".single-content",
        "article .content",
        ".article-body",
        "article",
    ])


# ── Gorkhapatra ───────────────────────────────────────────────────────────────

def get_gorkhapatra_urls(max_urls=2000):
    """Get news article URLs from gorkhapatra sitemap."""
    urls = []
    for sitemap_name in ["sitemap-news.xml", "sitemap-categories.xml"]:
        r = fetch(f"https://gorkhapatra.org.np/{sitemap_name}")
        if not r:
            continue
        locs = re.findall(r"<loc>(.*?)</loc>", r)
        # Only keep article URLs
        article_urls = [l for l in locs if "/content/" in l]
        urls.extend(article_urls)
        print(f"    {sitemap_name}: {len(article_urls)} article URLs")
        time.sleep(0.5)
    return list(dict.fromkeys(urls))[:max_urls]


def scrape_gorkhapatra(url):
    html = fetch(url)
    if not html:
        return None
    return extract_text(html, [
        ".jeg_post_content",
        ".content-inner",
        ".entry-content",
        ".post-body",
        "article",
    ])


# ── Main ──────────────────────────────────────────────────────────────────────

def scrape_source(name, urls_fn, scrape_fn, out_file, delay=0.4):
    print(f"\nFetching URLs for {name}...")
    urls = urls_fn()
    if not urls:
        print(f"  No URLs for {name}.")
        return 0

    saved = 0
    with open(out_file, "a", encoding="utf-8") as f:
        for i, url in enumerate(urls):
            text = scrape_fn(url)
            if text:
                f.write(text + "\n\n")
                saved += 1
            if (i + 1) % 100 == 0:
                size = Path(out_file).stat().st_size
                print(f"  [{name}] {i+1}/{len(urls)} | saved={saved} | ~{size//3:,} tokens")
            time.sleep(delay)

    size = Path(out_file).stat().st_size
    print(f"  [{name}] Done. {saved}/{len(urls)} | ~{size//3:,} tokens total")
    return saved


if __name__ == "__main__":
    out_path = "data/nepali/raw/manual_scraped.txt"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    # Start fresh (previous run produced 0 bytes)
    open(out_path, "w").close()

    print("=== Nepal News ===")
    scrape_source("NepaliNews", get_nepalnews_urls, scrape_nepalnews, out_path, delay=0.3)

    print("\n=== Gorkhapatra ===")
    scrape_source("Gorkhapatra", get_gorkhapatra_urls, scrape_gorkhapatra, out_path, delay=0.4)

    final = Path(out_path).stat().st_size
    print(f"\nTotal: {final/1e6:.1f}MB | ~{final//3:,} tokens -> {out_path}")
