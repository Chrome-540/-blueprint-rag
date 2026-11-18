"""
LinkedIn latest-posts job digest.
Run manually:  python linkedin_digest.py            (posts from last 30 min)
               python linkedin_digest.py 180        (posts from last 180 min)
First run: browser opens, log in to LinkedIn once, then press Enter in the terminal.
"""
import json, random, re, sys, time, webbrowser
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from playwright.sync_api import sync_playwright

# ---------------- CONFIG ----------------
MAX_AGE_MIN = int(sys.argv[1]) if len(sys.argv) > 1 else 30
QUERY = '"AI Engineer" OR "GenAI Engineer" OR "Generative AI Engineer" OR "LLM Engineer" hiring'
SCROLLS = 6                  # how many times to scroll the results
MAX_EXP_YEARS = 3            # keep posts whose minimum experience is below this
KEEP_NO_EXP_MENTION = True   # keep posts that don't state experience at all
REQUIRE_INDIA = True
REQUIRE_EMAIL = False        # True = only posts that contain an email id

BASE = Path(__file__).parent
PROFILE_DIR = BASE / "li_profile"             # keeps you logged in
SEEN_FILE = BASE / "seen.json"
# ----------------------------------------

INDIA_RE = re.compile(
    r"\b(india|hyderabad|bangalore|bengaluru|pune|chennai|mumbai|delhi|ncr|gurgaon|gurugram|"
    r"noida|kolkata|ahmedabad|kochi|coimbatore|indore|jaipur|trivandrum|lpa|₹)\b", re.I)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
EXP_RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)", re.I)
EXP_PLUS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)", re.I)
FRESHER_RE = re.compile(r"\b(fresher|freshers|entry[- ]level|0\s*(?:-|to)\s*\d)\b", re.I)
FAKE_MAIL_DOMAINS = ("linkedin.com", "example.com")


def post_time_from_urn(urn_id: int) -> datetime:
    """LinkedIn activity IDs encode the creation time: first 41 bits = ms since epoch."""
    ms = urn_id >> 22
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def min_experience(text: str):
    if FRESHER_RE.search(text):
        return 0.0
    m = EXP_RANGE_RE.search(text)
    if m:
        return float(m.group(1))
    m = EXP_PLUS_RE.search(text)
    if m:
        return float(m.group(1))
    return None


def extract(text: str) -> dict:
    emails = [e for e in EMAIL_RE.findall(text) if not e.lower().endswith(FAKE_MAIL_DOMAINS)]
    exp = min_experience(text)
    company = None
    m = re.search(r"(?:hiring at|join|company|at)\s*[:\-]?\s*([A-Z][\w&.\- ]{2,40})", text)
    if m:
        company = m.group(1).strip()
    return {"emails": sorted(set(emails)), "min_exp": exp, "company": company,
            "india": bool(INDIA_RE.search(text))}


BLOCK_KEYS = ("checkpoint", "/login", "authwall", "/uas/")


def wait_until_clear(page, timeout_s=300):
    """If LinkedIn shows login / verification, let the user finish it by hand."""
    start = time.time()
    warned = False
    while any(k in page.url for k in BLOCK_KEYS):
        if not warned:
            print("LinkedIn needs login/verification. Complete it in the Chrome window "
                  "(script waits up to 5 min)...")
            warned = True
        if time.time() - start > timeout_s:
            raise SystemExit("Timed out waiting for LinkedIn login/verification.")
        page.wait_for_timeout(2000)


def collect_posts(page):
    url = ("https://www.linkedin.com/search/results/content/?keywords="
           + quote(QUERY) + '&sortBy=%22date_posted%22&origin=FACETED_SEARCH')
    try:
        page.goto(url)
    except Exception:
        pass  # interrupted by a redirect to login/verification; handled below
    page.wait_for_timeout(3000)
    if any(k in page.url for k in BLOCK_KEYS):
        wait_until_clear(page)
        page.goto(url)
    page.wait_for_timeout(5000)
    for _ in range(SCROLLS):
        page.mouse.wheel(0, 3000)
        page.wait_for_timeout(random.randint(2500, 5000))
    return page.evaluate("""() => {
        const out = [];
        const seen = new Set();
        document.querySelectorAll('[data-urn*="urn:li:activity"], [data-chameleon-result-urn*="urn:li:activity"]')
          .forEach(el => {
            const urn = el.getAttribute('data-urn') || el.getAttribute('data-chameleon-result-urn');
            const m = urn && urn.match(/urn:li:activity:(\\d+)/);
            if (!m || seen.has(m[1])) return;
            seen.add(m[1]);
            out.push({id: m[1], text: el.innerText});
          });
        return out;
    }""")


def build_html(rows):
    parts = [f"<h3>{len(rows)} new posts (last {MAX_AGE_MIN} min window)</h3>"]
    for r in rows:
        exp = "not stated" if r["min_exp"] is None else f'{r["min_exp"]:g}+ yrs'
        mails = ", ".join(r["emails"]) or "none in post"
        snippet = re.sub(r"\s+", " ", r["text"])[:400]
        parts.append(
            f'<p><b>{r["age_min"]} min ago</b> | Exp: {exp} | Email: <b>{mails}</b><br>'
            f'{snippet}...<br><a href="{r["link"]}">Open post</a></p><hr>')
    return "".join(parts)


def main():
    seen = set(json.loads(SEEN_FILE.read_text())) if SEEN_FILE.exists() else set()
    now = datetime.now(timezone.utc)

    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            str(PROFILE_DIR), channel="chrome", headless=False,
            args=["--disable-blink-features=AutomationControlled"],
            ignore_default_args=["--enable-automation"])
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto("https://www.linkedin.com/feed/")
        page.wait_for_timeout(3000)
        wait_until_clear(page)
        posts = collect_posts(page)
        ctx.close()

    rows = []
    for post in posts:
        pid = post["id"]
        if pid in seen:
            continue
        age_min = round((now - post_time_from_urn(int(pid))).total_seconds() / 60)
        if age_min > MAX_AGE_MIN or age_min < 0:
            continue
        info = extract(post["text"])
        if REQUIRE_INDIA and not info["india"]:
            continue
        if info["min_exp"] is None and not KEEP_NO_EXP_MENTION:
            continue
        if info["min_exp"] is not None and info["min_exp"] >= MAX_EXP_YEARS:
            continue
        if REQUIRE_EMAIL and not info["emails"]:
            continue
        rows.append({**info, "id": pid, "age_min": age_min, "text": post["text"],
                     "link": f"https://www.linkedin.com/feed/update/urn:li:activity:{pid}/"})

    # mark every scanned post as seen so it never repeats
    seen.update(p_["id"] for p_ in posts)
    SEEN_FILE.write_text(json.dumps(sorted(seen)))

    rows.sort(key=lambda r: (not r["emails"], r["age_min"]))  # posts with email first, then newest
    print(f"Scanned {len(posts)} posts, {len(rows)} matched.")
    if not rows:
        return
    out = BASE / "last_digest.html"
    out.write_text(build_html(rows), encoding="utf-8")
    print(f"Saved {out.name}, opening in browser.")
    webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()