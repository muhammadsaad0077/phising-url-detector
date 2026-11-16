import re
from urllib.parse import urlparse

# Words commonly seen in phishing URLs (login pages, fake security alerts, etc.)
SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "verification", "account", "secure",
    "security", "update", "confirm", "banking", "bank", "password",
    "pay", "billing", "wallet", "webscr", "ebayisapi", "suspend",
    "limited", "urgent", "click", "free", "bonus",
]

IP_PATTERN = re.compile(
    r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$"
)

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "adf.ly", "shorte.st", "cutt.ly", "rebrand.ly",
}

# The exact, ordered list of feature names the model is trained on.
FEATURE_NAMES = [
    "url_length",
    "domain_length",
    "path_length",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_slashes",
    "num_digits",
    "num_letters",
    "num_equal",
    "num_qmark",
    "num_ampersand",
    "num_at_symbol",
    "num_percent",
    "num_special_chars",
    "digit_ratio",
    "letter_ratio",
    "num_subdomains",
    "has_https",
    "has_ip_address",
    "has_www",
    "has_port",
    "tld_length",
    "num_suspicious_words",
    "is_shortened",
]


def _normalize(url: str) -> str:
    """Ensure the URL has a scheme so urlparse behaves consistently."""
    url = url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url
    return url


def extract_url_features(raw_url: str) -> dict:
    """Given a raw URL string, return a dict of {feature_name: value}."""
    url = _normalize(raw_url)
    parsed = urlparse(url)

    domain = parsed.netloc.split("@")[-1]  # strip user:pass@ if present
    domain_no_port = domain.split(":")[0]
    path = parsed.path or ""

    url_len = len(raw_url)
    domain_len = len(domain_no_port)
    path_len = len(path)

    num_dots = raw_url.count(".")
    num_hyphens = raw_url.count("-")
    num_underscores = raw_url.count("_")
    num_slashes = raw_url.count("/")
    num_digits = sum(c.isdigit() for c in raw_url)
    num_letters = sum(c.isalpha() for c in raw_url)
    num_equal = raw_url.count("=")
    num_qmark = raw_url.count("?")
    num_ampersand = raw_url.count("&")
    num_at_symbol = raw_url.count("@")
    num_percent = raw_url.count("%")

    known_chars = num_dots + num_hyphens + num_underscores + num_slashes + \
        num_digits + num_letters + num_equal + num_qmark + \
        num_ampersand + num_at_symbol + num_percent
    num_special_chars = max(len(raw_url) - known_chars, 0)

    digit_ratio = num_digits / url_len if url_len else 0.0
    letter_ratio = num_letters / url_len if url_len else 0.0

    domain_parts = domain_no_port.split(".") if domain_no_port else []
    # e.g. "login.accounts.example.co.uk" -> subdomains beyond domain+tld
    num_subdomains = max(len(domain_parts) - 2, 0)

    has_https = 1 if parsed.scheme == "https" else 0
    has_ip_address = 1 if IP_PATTERN.match(domain_no_port) else 0
    has_www = 1 if domain_no_port.startswith("www.") else 0
    has_port = 1 if (":" in domain and domain.split(":")[-1].isdigit()) else 0

    tld = domain_parts[-1] if domain_parts else ""
    tld_length = len(tld)

    lower_url = raw_url.lower()
    num_suspicious_words = sum(1 for w in SUSPICIOUS_WORDS if w in lower_url)

    is_shortened = 1 if domain_no_port.lower() in SHORTENERS else 0

    return {
        "url_length": url_len,
        "domain_length": domain_len,
        "path_length": path_len,
        "num_dots": num_dots,
        "num_hyphens": num_hyphens,
        "num_underscores": num_underscores,
        "num_slashes": num_slashes,
        "num_digits": num_digits,
        "num_letters": num_letters,
        "num_equal": num_equal,
        "num_qmark": num_qmark,
        "num_ampersand": num_ampersand,
        "num_at_symbol": num_at_symbol,
        "num_percent": num_percent,
        "num_special_chars": num_special_chars,
        "digit_ratio": round(digit_ratio, 4),
        "letter_ratio": round(letter_ratio, 4),
        "num_subdomains": num_subdomains,
        "has_https": has_https,
        "has_ip_address": has_ip_address,
        "has_www": has_www,
        "has_port": has_port,
        "tld_length": tld_length,
        "num_suspicious_words": num_suspicious_words,
        "is_shortened": is_shortened,
    }


if __name__ == "__main__":
    # quick manual sanity check
    test_urls = [
        "http://login-account-security-example.com/verify?id=12345",
        "https://www.google.com",
        "http://192.168.1.1/wp-admin/verify.php",
    ]
    for u in test_urls:
        print(u)
        print(extract_url_features(u))
        print()