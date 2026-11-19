"""Break the model on purpose (do not remove the mover from its own zone's load) and count failing tests. -> results/mutation_check.txt"""
import re, shutil, subprocess, sys
src = "game/model.py"; shutil.copy(src, "/tmp/model.orig")
try:
    s = open(src).read(); assert "n_z[a[i]] -= 1.0" in s
    open(src, "w").write(s.replace("n_z[a[i]] -= 1.0          # remove i from its current zone", "pass"))
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_game.py"], capture_output=True, text=True).stdout
finally:
    shutil.copy("/tmp/model.orig", src)
m = re.search(r"(\d+) failed", out); k = int(m.group(1)) if m else 0
open("results/mutation_check.txt", "w").write(f"failed_tests={k}\n"); print("failed", k)
