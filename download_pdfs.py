import requests
import os
import urllib3

urllib3.disable_warnings()

pdfs = {
    "anti-terrorism-act-1997.pdf": [
        "https://na.gov.pk/uploads/documents/1549887551_526.pdf",
        "http://www.na.gov.pk/uploads/documents/1326710688_680.pdf",
        "https://www.fmu.gov.pk/docs/laws/Anti-Terrorism_Act_1997.pdf"
    ],
    "control-of-narcotic-substances-act-1997.pdf": [
        "http://na.gov.pk/uploads/documents/1324446450_558.pdf", 
        "https://www.fmu.gov.pk/docs/laws/Control_of_Narcotic_Substances_Act_1997.pdf"
    ]
}

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

output_dir = "backend/data/pdfs"

for filename, urls in pdfs.items():
    success = False
    for url in urls:
        print(f"Trying to download {filename} from {url}...")
        try:
            r = requests.get(url, headers=headers, timeout=20, verify=False, allow_redirects=True)
            if r.status_code == 200 and len(r.content) > 10000: # Ensure > 10KB
                with open(os.path.join(output_dir, filename), "wb") as f:
                    f.write(r.content)
                print(f"SUCCESS: Downloaded {filename} ({len(r.content)} bytes)")
                success = True
                break
            else:
                print(f"FAILED: Status {r.status_code}, Length {len(r.content)}")
        except Exception as e:
            print(f"ERROR: {e}")
    
    if not success:
        print(f"CRITICAL: All mirrors failed for {filename}")
