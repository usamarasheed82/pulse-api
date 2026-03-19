"""
api/scan.py  —  Vercel Serverless Function
Deployed at: https://pulse-api.vercel.app/api/scan

Vercel runs this as a serverless Python function.
No server to manage. Free tier handles ~100k requests/month.
"""

import asyncio, hashlib, json, os, re, time, uuid
from datetime import datetime
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from http.server import BaseHTTPRequestHandler

# ─── Signal weights ────────────────────────────────────────────────────────
SIGNALS = {
    "agents_json":    {"w": 20, "name": "agents.json"},
    "llms_txt":       {"w": 15, "name": "llms.txt"},
    "product_schema": {"w": 15, "name": "Product JSON-LD schema"},
    "open_graph":     {"w": 8,  "name": "Open Graph tags"},
    "product_desc":   {"w": 8,  "name": "Product descriptions"},
    "breadcrumbs":    {"w": 7,  "name": "Breadcrumb schema"},
    "sitemap":        {"w": 7,  "name": "Sitemap XML"},
    "robots_ai":      {"w": 5,  "name": "robots.txt AI rules"},
    "review_schema":  {"w": 5,  "name": "Review schema"},
    "canonical":      {"w": 5,  "name": "Canonical URLs"},
    "faq_schema":     {"w": 3,  "name": "FAQ schema"},
    "https_check":    {"w": 2,  "name": "HTTPS"},
}

HEADERS = {
    "User-Agent": "AgenticPulseScorer/1.0 Mozilla/5.0 compatible",
    "Accept": "text/html,application/json,*/*",
}

# ─── Scanner ───────────────────────────────────────────────────────────────
async def scan_url(url: str) -> dict:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parsed   = urlparse(url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    domain   = parsed.netloc.replace("www.", "")
    results  = []

    async with httpx.AsyncClient(
        headers=HEADERS, timeout=12,
        follow_redirects=True, verify=False
    ) as client:
        # Fetch homepage once — many checks reuse it
        html = ""
        soup = None
        try:
            r = await client.get(base_url)
            if r.status_code == 200:
                html = r.text
                soup = BeautifulSoup(html, "html.parser")
        except Exception:
            pass

        # Run all checks concurrently
        checks = await asyncio.gather(
            check_agents_json(client, base_url),
            check_llms_txt(client, base_url),
            check_product_schema(soup),
            check_open_graph(soup),
            check_product_desc(client, base_url, soup),
            check_breadcrumbs(soup),
            check_sitemap(client, base_url),
            check_robots(client, base_url),
            check_review_schema(soup),
            check_canonical(soup),
            check_faq_schema(soup),
            check_https(parsed),
        )
        results = list(checks)

    total   = sum(r["score"] for r in results)
    tier, color = get_tier(total)
    failed  = sorted([r for r in results if not r["passed"]], key=lambda x: -x["weight"])
    top3    = [f"{r['name']}: {r['fix']}" for r in failed[:3]]

    return {
        "id":       str(uuid.uuid4()),
        "domain":   domain,
        "score":    total,
        "tier":     tier,
        "color":    color,
        "impact":   get_impact(total),
        "top_gaps": top3,
        "signals":  results,
        "preview":  _preview(results),
        "scanned_at": datetime.utcnow().isoformat(),
    }


def _preview(results):
    """Top 2 failures + 1 pass — shown before email gate."""
    failed = sorted([r for r in results if not r["passed"]], key=lambda x: -x["weight"])
    passed = [r for r in results if r["passed"]]
    items  = failed[:2] + passed[:1]
    return [{"name": r["name"], "passed": r["passed"],
             "weight": r["weight"], "detail": r["detail"]} for r in items]


# ─── Individual checks ─────────────────────────────────────────────────────
async def check_agents_json(client, base):
    key, w = "agents_json", 20
    try:
        r = await client.get(urljoin(base, "/agents.json"))
        if r.status_code == 200:
            d = r.json()
            if ("name" in d or "store_name" in d) and \
               any(k in d for k in ("products","catalog","capabilities")):
                return sig(key, w, True, w,  "Valid agents.json found", "Already done")
            return sig(key, w, False, w//2, "agents.json exists but incomplete",
                       "Add name, capabilities, and products fields")
    except Exception:
        pass
    return sig(key, w, False, 0,
               "agents.json missing — AI agents can't discover your store",
               "Create /agents.json with store name, description, and product catalog")


async def check_llms_txt(client, base):
    key, w = "llms_txt", 15
    for path in ("/llms.txt", "/llms-full.txt"):
        try:
            r = await client.get(urljoin(base, path))
            if r.status_code == 200 and len(r.text) > 50:
                return sig(key, w, True, w, f"llms.txt found ({len(r.text)} chars)", "Already done")
        except Exception:
            pass
    return sig(key, w, False, 0,
               "llms.txt missing — LLMs have no context about your store",
               "Create /llms.txt with brand description, product categories, and policies")


async def check_product_schema(soup):
    key, w = "product_schema", 15
    if not soup:
        return sig(key, w, False, 0, "Could not fetch page", "Ensure homepage is public")
    schemas = get_jsonld(soup)
    hits = [s for s in schemas if s.get("@type") in ("Product", "ItemList")]
    if hits:
        return sig(key, w, True, w, f"{len(hits)} Product schema(s) found", "Already done")
    return sig(key, w, False, 0,
               "No Product JSON-LD — AI can't parse your product details",
               "Add JSON-LD Product schema with name, price, description, image to product pages")


async def check_open_graph(soup):
    key, w = "open_graph", 8
    if not soup:
        return sig(key, w, False, 0, "Page inaccessible", "Add OG tags")
    required = {"og:title","og:description","og:image","og:url"}
    found    = {t.get("property") for t in soup.find_all("meta", property=re.compile(r"^og:"))}
    missing  = required - found
    if not missing:
        return sig(key, w, True, w, "All OG tags present", "Already done")
    partial = int(w * len(found & required) / len(required))
    return sig(key, w, len(missing) <= 1, partial,
               f"Missing: {', '.join(missing)}",
               f"Add meta property tags: {', '.join(missing)}")


async def check_product_desc(client, base, soup):
    key, w = "product_desc", 8
    if soup:
        links = [a["href"] for a in soup.find_all("a", href=True)
                 if any(k in a["href"].lower() for k in ("/products/","/p/","/item/"))]
        if links:
            try:
                r = await client.get(urljoin(base, links[0]))
                if r.status_code == 200:
                    ps = BeautifulSoup(r.text, "html.parser")
                    for sel in [".product-description","[itemprop='description']",
                                 ".description","#product-description"]:
                        el = ps.select_one(sel)
                        if el and len(el.get_text(strip=True)) > 150:
                            return sig(key, w, True, w,
                                       "Rich product descriptions found", "Already done")
                    return sig(key, w, False, w//2,
                               "Product descriptions are thin (<150 chars)",
                               "Expand descriptions to 200+ words with use cases and specs")
            except Exception:
                pass
    return sig(key, w, False, 0,
               "Could not evaluate product descriptions",
               "Ensure product pages are public with detailed descriptions")


async def check_breadcrumbs(soup):
    key, w = "breadcrumbs", 7
    if soup:
        schemas = get_jsonld(soup)
        if any(s.get("@type") == "BreadcrumbList" for s in schemas):
            return sig(key, w, True, w, "BreadcrumbList schema found", "Already done")
    return sig(key, w, False, 0,
               "No breadcrumb schema — AI can't understand your site structure",
               "Add BreadcrumbList JSON-LD to category and product pages")


async def check_sitemap(client, base):
    key, w = "sitemap", 7
    for path in ("/sitemap.xml","/sitemap_index.xml","/sitemap.txt"):
        try:
            r = await client.get(urljoin(base, path))
            if r.status_code == 200:
                return sig(key, w, True, w, f"Sitemap at {path}", "Already done")
        except Exception:
            pass
    return sig(key, w, False, 0,
               "No sitemap.xml — AI crawlers can't discover your full catalogue",
               "Generate sitemap.xml and submit to search/AI crawlers")


async def check_robots(client, base):
    key, w = "robots_ai", 5
    try:
        r = await client.get(urljoin(base, "/robots.txt"))
        if r.status_code == 200:
            content = r.text.lower()
            ai_bots = ["gptbot","claudebot","perplexitybot","googlebot"]
            if any(b in content for b in ai_bots):
                return sig(key, w, True, w, "robots.txt mentions AI crawlers", "Already done")
            return sig(key, w, False, w//2, "robots.txt exists but no AI crawler rules",
                       "Add Allow rules for GPTBot, ClaudeBot, PerplexityBot")
    except Exception:
        pass
    return sig(key, w, False, 0,
               "No robots.txt — AI crawler behaviour undefined",
               "Create robots.txt allowing major AI shopping crawlers")


async def check_review_schema(soup):
    key, w = "review_schema", 5
    if soup:
        schemas = get_jsonld(soup)
        if any(s.get("@type") in ("AggregateRating","Review") or
               "aggregateRating" in s for s in schemas):
            return sig(key, w, True, w, "AggregateRating schema found", "Already done")
    return sig(key, w, False, 0,
               "No review schema — AI can't surface your ratings",
               "Add AggregateRating JSON-LD with ratingValue and reviewCount")


async def check_canonical(soup):
    key, w = "canonical", 5
    if soup and soup.find("link", rel="canonical", href=True):
        return sig(key, w, True, w, "Canonical tag found", "Already done")
    return sig(key, w, False, 0,
               "No canonical URL — AI may index duplicate content",
               "Add <link rel='canonical'> to every page")


async def check_faq_schema(soup):
    key, w = "faq_schema", 3
    if soup:
        schemas = get_jsonld(soup)
        if any(s.get("@type") == "FAQPage" for s in schemas):
            return sig(key, w, True, w, "FAQPage schema found", "Already done")
    return sig(key, w, False, 0,
               "No FAQ schema — missed Q&A opportunity for AI",
               "Add FAQPage JSON-LD with common product questions")


async def check_https(parsed):
    key, w = "https_check", 2
    if parsed.scheme == "https":
        return sig(key, w, True, w, "HTTPS enforced", "Already done")
    return sig(key, w, False, 0, "Not using HTTPS",
               "Enable SSL — free via Let's Encrypt or your platform")


# ─── Helpers ───────────────────────────────────────────────────────────────
def sig(key, weight, passed, score, detail, fix):
    return {"key": key, "name": SIGNALS[key]["name"], "weight": weight,
            "passed": passed, "score": score, "detail": detail, "fix": fix}

def get_jsonld(soup) -> list:
    out = []
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            d = json.loads(tag.string or "")
            if isinstance(d, list): out.extend(d)
            elif isinstance(d, dict):
                out.extend(d.get("@graph", [d]))
        except Exception:
            pass
    return out

def get_tier(score):
    if score >= 80: return "optimised",   "#085041"
    if score >= 55: return "capable",     "#0F6E56"
    if score >= 30: return "emerging",    "#854F0B"
    return             "agent-blind", "#791F1F"

def get_impact(score):
    if score >= 80: return "Great — your store has strong AI visibility"
    if score >= 55: return "Moderate — some gaps worth addressing"
    if score >= 30: return "High — significant AI revenue being missed"
    return              "Critical — store is largely invisible to AI assistants"


# ─── Vercel handler ────────────────────────────────────────────────────────
class handler(BaseHTTPRequestHandler):

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_POST(self):
        length  = int(self.headers.get("Content-Length", 0))
        body    = self.rfile.read(length)
        try:
            data = json.loads(body)
            url  = data.get("url", "").strip()
            if not url:
                raise ValueError("url required")
        except Exception as e:
            self._json({"error": str(e)}, 400)
            return

        try:
            result = asyncio.run(scan_url(url))
            self._json(result, 200)
        except Exception as e:
            self._json({"error": f"Scan failed: {str(e)}"}, 422)

    def _json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, *args):
        pass  # Suppress Vercel logs spam
