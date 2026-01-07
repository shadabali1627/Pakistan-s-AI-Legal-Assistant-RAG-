import requests
import os
import urllib3

urllib3.disable_warnings()

# Targeted, high-probability links
targets = [
    {
        "filename": "anti-terrorism-act-1997.pdf",
        "url": "https://www.unodc.org/tldb/pdf/Pakistan_Anti-Terrorism_Act_1997.pdf"
    },
    {
        "filename": "anti-terrorism-act-1997.pdf", # Backup
        "url": "https://na.gov.pk/uploads/documents/1549887551_526.pdf"
    },
    {
        "filename": "control-of-narcotic-substances-act-1997.pdf",
        "url": "https://na.gov.pk/uploads/documents/1324446450_558.pdf"
    }
]

headers = {"User-Agent": "Mozilla/5.0"}
output_dir = "backend/data/pdfs"

for target in targets:
    fname = target["filename"]
    fpath = os.path.join(output_dir, fname)
    
    # Skip if we already successfully downloaded it (size > 10KB)
    if os.path.exists(fpath) and os.path.getsize(fpath) > 10000:
        print(f"SKIP: {fname} already valid.")
        continue

    print(f"Downloading {fname} from {target['url']}...")
    try:
        r = requests.get(target["url"], headers=headers, verify=False, timeout=30)
        if r.status_code == 200 and len(r.content) > 10000:
            with open(fpath, "wb") as f:
                f.write(r.content)
            print(f"SUCCESS: {fname} downloaded ({len(r.content)} bytes).")
        else:
            print(f"FAIL: {fname} -> Status {r.status_code}, Len {len(r.content)}")
    except Exception as e:
        print(f"ERROR: {fname} -> {e}")
