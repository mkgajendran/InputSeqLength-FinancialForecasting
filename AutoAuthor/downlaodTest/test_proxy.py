import requests

proxies = {
    "http": "http://package-299236-opt-wb:2q6zkzD6WNmTZV32@proxy.soax.com:5000",
    "https": "http://package-299236-opt-wb:2q6zkzD6WNmTZV32@proxy.soax.com:5000"
}

try:
    response = requests.get("https://www.google.com", proxies=proxies, timeout=10)
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.text[:200]}")
except Exception as e:
    print(f"Proxy test failed: {e}")