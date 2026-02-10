import subprocess, sys

scripts = [
    "scripts/eda_01_hahackathon.py",
    "scripts/eda_02_detection.py",
    "scripts/eda_03_reddit.py",
]

for s in scripts:
    print("\nRunning:", s)
    r = subprocess.run([sys.executable, s])
    if r.returncode != 0:
        raise SystemExit(f"Failed: {s}")
print("\nALL DONE.")
