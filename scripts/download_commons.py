import os
import json
import time
from typing import List, Dict

import requests
from rich.console import Console


console = Console()


API = "https://commons.wikimedia.org/w/api.php"


def build_user_agent() -> str:
    # Wikimedia asks for descriptive UA with contact info
    return os.environ.get(
        "COMMONS_UA",
        "AnimeUpscaler/0.1 (local; set COMMONS_UA with contact info)",
    )


def query_images(limit: int = 40) -> List[Dict]:
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": "anime drawing|manga drawing|anime art|manga art",
        "gsrlimit": str(limit),
        "gsrnamespace": 6,  # File namespace
        "prop": "imageinfo|info|extmetadata",
        "iiprop": "url|size|mime",
        "format": "json",
    }
    headers = {"User-Agent": build_user_agent()}
    session = requests.Session()

    attempts = 0
    while True:
        resp = session.get(API, params=params, headers=headers, timeout=30)
        if resp.status_code in (403, 429):
            attempts += 1
            if attempts >= 3:
                console.print(f"[red]HTTP {resp.status_code} depuis Wikimedia.[/red]")
                console.print("[yellow]Astuce:[/yellow] définissez COMMONS_UA avec un User-Agent descriptif incluant un moyen de contact, puis réessayez.")
                console.print("Ex: $env:COMMONS_UA=\"AnimeUpscaler/0.1 (Windows; contact: email@example.com)\"")
                resp.raise_for_status()
            wait_s = 2 ** attempts
            console.print(f"HTTP {resp.status_code} — nouvelle tentative dans {wait_s}s...")
            time.sleep(wait_s)
            continue
        resp.raise_for_status()
        data = resp.json()
        break
    pages = data.get("query", {}).get("pages", {})
    results = []
    for _, page in pages.items():
        if "imageinfo" not in page:
            continue
        info = page["imageinfo"][0]
        meta = page.get("extmetadata", {})
        lic = (meta.get("LicenseShortName", {}).get("value") or "").lower()
        if not ("cc" in lic or "public" in lic):
            continue
        url = info.get("url")
        if not url:
            continue
        if not any(url.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp"]):
            continue
        results.append({
            "title": page.get("title"),
            "url": url,
            "license": lic,
            "author": meta.get("Artist", {}).get("value"),
            "credit": meta.get("Credit", {}).get("value"),
            "source": meta.get("AttributionRequired", {}).get("value"),
            "description": meta.get("ImageDescription", {}).get("value"),
            "raw_meta": meta,
        })
    return results


def download(url: str, path: str) -> None:
    resp = requests.get(url, headers={"User-Agent": build_user_agent()}, timeout=30)
    resp.raise_for_status()
    with open(path, "wb") as f:
        f.write(resp.content)


def main() -> None:
    out_dir = os.path.join("data", "commons", "images")
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join("data", "commons", "metadata.json")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)

    console.print("Recherche d'images sur Wikimedia Commons...")
    items = query_images(limit=60)
    if not items:
        console.print("[red]Aucune image trouvée.[/red]")
        return

    saved = []
    for it in items:
        name = it["title"].replace("/", "_").replace(":", "_")
        ext = os.path.splitext(it["url"])[1].lower()
        out = os.path.join(out_dir, name + ext)
        try:
            download(it["url"], out)
            saved.append({**it, "local_path": out})
            console.print(f"[green]Téléchargé:[/green] {out}")
        except Exception as e:
            console.print(f"[yellow]Échec:[/yellow] {it['url']} ({e})")
        time.sleep(0.2)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"items": saved}, f, ensure_ascii=False, indent=2)
    console.print(f"Métadonnées: {meta_path}")

    # Optionnel: copier vers data/hr pour entraînement
    hr_dir = os.path.join("data", "hr")
    os.makedirs(hr_dir, exist_ok=True)
    copied = 0
    import shutil
    for it in saved:
        src = it["local_path"]
        dst = os.path.join(hr_dir, os.path.basename(src))
        try:
            shutil.copy2(src, dst)
            copied += 1
        except Exception:
            pass
    console.print(f"Copiées vers data/hr: {copied}")


if __name__ == "__main__":
    main()


