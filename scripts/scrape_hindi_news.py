"""
Hindi news scraper: prabhat khabar, BBC Hindi, Dainik Jagran, NDTV Hindi
Target: manual Hindi tokens for 20% requirement.
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
    "Accept-Language": "hi-IN,hi;q=0.9,en-US;q=0.8",
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


def extract_article_text(html, selectors):
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
    paras = soup.find_all("p")
    text = "\n".join(p.get_text(" ", strip=True) for p in paras if len(p.get_text()) > 30)
    text = clean(text)
    return text if has_devanagari(text) and len(text) > 100 else None


def urls_from_sitemap(url, filter_fn=None, max_urls=5000, depth=0):
    if depth > 3:
        return []
    html = fetch(url)
    if not html:
        return []
    urls = []
    try:
        html_clean = re.sub(r'<\?xml[^>]*\?>', '', html).strip()
        root = ET.fromstring(html_clean)
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        for sitemap in root.findall("sm:sitemap/sm:loc", ns):
            if len(urls) >= max_urls:
                break
            child = urls_from_sitemap(sitemap.text.strip(), filter_fn, max_urls - len(urls), depth + 1)
            urls.extend(child)
            time.sleep(0.3)
        for url_el in root.findall("sm:url/sm:loc", ns):
            loc = url_el.text.strip()
            if filter_fn is None or filter_fn(loc):
                urls.append(loc)
            if len(urls) >= max_urls:
                break
    except ET.ParseError:
        pass
    return urls


# ── Prabhat Khabar ────────────────────────────────────────────────────────────

def get_prabhatkhabar_urls(max_urls=5000):
    print("  Fetching Prabhat Khabar sitemap...")
    urls = urls_from_sitemap(
        "https://www.prabhatkhabar.com/sitemap.xml",
        filter_fn=lambda u: "prabhatkhabar.com" in u and len(u) > 35,
        max_urls=max_urls
    )
    print(f"  Prabhat Khabar: {len(urls)} URLs")
    return urls


def scrape_prabhatkhabar(url):
    html = fetch(url)
    if not html:
        return None
    return extract_article_text(html, [
        ".article-full-content",
        ".story-content",
        ".post-content",
        "article .content",
        "article",
        ".main-content",
    ])


# ── BBC Hindi ────────────────────────────────────────────────────────────────

def get_bbc_hindi_urls(max_urls=3000):
    print("  Fetching BBC Hindi sitemap...")
    urls = urls_from_sitemap(
        "https://www.bbc.com/hindi/sitemap.xml",
        filter_fn=lambda u: "bbc.com/hindi" in u and len(u) > 30,
        max_urls=max_urls
    )
    if len(urls) < 50:
        urls = urls_from_sitemap(
            "https://www.bbc.com/sitemaps/https-sitemap-com-hindi-1.xml",
            max_urls=max_urls
        )
    print(f"  BBC Hindi: {len(urls)} URLs")
    return urls


def scrape_bbc_hindi(url):
    html = fetch(url)
    if not html:
        return None
    return extract_article_text(html, [
        "[data-component='text-block']",
        ".story-body__inner",
        ".article__body-content",
        "article",
        "[data-e2e='story-body']",
        ".bbc-19j92fr",
    ])


# ── NDTV Hindi ────────────────────────────────────────────────────────────────

def get_ndtv_hindi_urls(max_urls=3000):
    print("  Fetching NDTV Hindi sitemap...")
    urls = urls_from_sitemap(
        "https://hindi.ndtv.com/sitemaps/sitemap-index.xml",
        filter_fn=lambda u: "hindi.ndtv.com" in u and len(u) > 35,
        max_urls=max_urls
    )
    print(f"  NDTV Hindi: {len(urls)} URLs")
    return urls


def scrape_ndtv_hindi(url):
    html = fetch(url)
    if not html:
        return None
    return extract_article_text(html, [
        ".sp-cn.ins_storybody",
        ".story__content",
        ".content_text",
        "article .article-content",
        ".post_content",
        "article",
    ])


# ── Dainik Jagran ─────────────────────────────────────────────────────────────

def get_jagran_urls(max_urls=5000):
    print("  Fetching Dainik Jagran sitemap...")
    urls = urls_from_sitemap(
        "https://www.jagran.com/sitemap.xml",
        filter_fn=lambda u: "jagran.com" in u and len(u) > 35,
        max_urls=max_urls
    )
    print(f"  Jagran: {len(urls)} URLs")
    return urls


def scrape_jagran(url):
    html = fetch(url)
    if not html:
        return None
    return extract_article_text(html, [
        ".article-content",
        ".detail-content",
        "#storyContent",
        ".news-content",
        "article",
    ])


# ── Main ──────────────────────────────────────────────────────────────────────

def scrape_source(name, get_urls_fn, scrape_fn, out_file, delay=0.4, max_urls=5000):
    urls = get_urls_fn()
    urls = urls[:max_urls]
    if not urls:
        print(f"  No URLs for {name}, skipping.")
        return 0

    saved = 0
    with open(out_file, "a", encoding="utf-8") as f:
        for i, url in enumerate(urls):
            text = scrape_fn(url)
            if text:
                f.write(text + "\n\n")
                saved += 1
            if (i + 1) % 200 == 0:
                size = Path(out_file).stat().st_size
                print(f"  [{name}] {i+1}/{len(urls)} | saved={saved} | ~{size//3:,} tokens")
            time.sleep(delay)

    size = Path(out_file).stat().st_size
    print(f"  [{name}] Done. {saved}/{len(urls)} | ~{size//3:,} tokens total")
    return saved


if __name__ == "__main__":
    out_path = "data/hindi/raw/manual_scraped.txt"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    open(out_path, "w").close()

    print("\n=== Prabhat Khabar ===")
    scrape_source("PrabhatKhabar", get_prabhatkhabar_urls, scrape_prabhatkhabar, out_path, delay=0.4, max_urls=5000)

    print("\n=== BBC Hindi ===")
    scrape_source("BBCHindi", get_bbc_hindi_urls, scrape_bbc_hindi, out_path, delay=0.5, max_urls=3000)

    print("\n=== NDTV Hindi ===")
    scrape_source("NDTVHindi", get_ndtv_hindi_urls, scrape_ndtv_hindi, out_path, delay=0.4, max_urls=3000)

    print("\n=== Dainik Jagran ===")
    scrape_source("Jagran", get_jagran_urls, scrape_jagran, out_path, delay=0.4, max_urls=5000)

    final = Path(out_path).stat().st_size
    print(f"\nTotal: {final/1e6:.1f}MB | ~{final//3:,} tokens -> {out_path}")
