"""Integrity list of every saved output (sha256, size, mtime, path) for the nightly NAS sync.

Deimos' SSD has silently zeroed files before (2026-10). A file whose size and mtime are unchanged since the last
list but whose hash differs was corrupted on disk, not rewritten: those are printed as CORRUPT (exit code 3).
Checkpoint directories are skipped (large, resume state only). Lists are kept in .hashes/ (one per run, none removed).
  python v3/hash_tree.py  > today.tsv
"""
import hashlib, os, sys, time
from pathlib import Path

BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent   # argv form: piped via stdin
DIRS = ["runs", "bench_base", "cmp", "speed", "code_snapshots", "v3"]
HD = BASE / ".hashes"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    HD.mkdir(exist_ok=True)
    prev = {}
    lists = sorted(HD.glob("hashes_*.tsv"))
    if lists:
        for line in lists[-1].open(encoding="utf-8"):
            s, size, mt, path = line.rstrip("\n").split("\t", 3)
            prev[path] = (s, size, mt)
    rows, corrupt = [], []
    for d in DIRS:
        for root, dirs, files in os.walk(BASE / d):
            dirs[:] = [x for x in dirs if not x.startswith("checkpoint-") and x != "__pycache__"]
            for fn in files:
                p = Path(root) / fn
                if p.is_symlink() or not p.is_file():
                    continue
                st = p.stat()
                rel = str(p.relative_to(BASE))
                size, mt = str(st.st_size), str(int(st.st_mtime))
                if time.time() - st.st_mtime < 120:
                    continue   # being written right now
                s = sha(p)
                rows.append(f"{s}\t{size}\t{mt}\t{rel}")
                old = prev.get(rel)
                if old and old[1] == size and old[2] == mt and old[0] != s:
                    corrupt.append(rel)
    out = HD / f"hashes_{time.strftime('%Y%m%d_%H%M%S')}.tsv"
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    sys.stdout.write(out.read_text(encoding="utf-8"))
    for c in corrupt:
        print(f"CORRUPT\t{c}", file=sys.stderr)
    print(f"[hash_tree] {len(rows)} files, {len(corrupt)} changed content with the same size and mtime -> {out.name}",
          file=sys.stderr)
    sys.exit(3 if corrupt else 0)


if __name__ == "__main__":
    main()
