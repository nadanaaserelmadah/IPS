"""Range-read only oxts/ and image_02/timestamps.txt from KITTI raw zips on the public S3 mirror."""
import io, ssl, sys, zipfile, urllib.request

import certifi

CTX = ssl.create_default_context(cafile=certifi.where())  # python.org builds on macOS ship no CA store
from pathlib import Path

BASE = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"
BLOCK = 4 << 20


class RangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos, self.cache, self.nreq, self.nbytes = url, 0, {}, 0, 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60, context=CTX) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def _block(self, b):
        if b not in self.cache:
            lo, hi = b * BLOCK, min(self.size, (b + 1) * BLOCK) - 1
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={lo}-{hi}"})
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(req, timeout=120, context=CTX) as r:
                        data = r.read()
                    break
                except Exception:
                    if attempt == 4:
                        raise
            self.nreq += 1; self.nbytes += len(data)
            if len(self.cache) > 64:
                self.cache.pop(next(iter(self.cache)))
            self.cache[b] = data
        return self.cache[b]

    def read(self, n=-1):
        if n < 0: n = self.size - self.pos
        out = bytearray()
        while n > 0 and self.pos < self.size:
            b = self.pos // BLOCK
            blk = self._block(b)
            o = self.pos - b * BLOCK
            chunk = blk[o:o + n]
            out += chunk; self.pos += len(chunk); n -= len(chunk)
        return bytes(out)

    def readinto(self, buf):
        d = self.read(len(buf)); buf[:len(d)] = d; return len(d)


def fetch(drive, kind, want, out: Path):
    url = f"{BASE}/{drive}/{drive}_{kind}.zip"
    f = RangeFile(url)
    z = zipfile.ZipFile(f)
    names = [n for n in z.namelist() if want(n) and not n.endswith("/")]
    names.sort(key=lambda n: z.getinfo(n).header_offset)
    for n in names:
        rel = n.split(f"{drive}_{kind}/", 1)[1]
        p = out / kind / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(z.read(n))
    print(f"{drive} {kind}: {len(names)} files, {f.nreq} requests, {f.nbytes / 1e6:.1f} MB read of {f.size / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    root = Path(sys.argv[1])
    for drive in sys.argv[2:]:
        out = root / drive
        fetch(drive, "extract", lambda n: "/oxts/" in n, out)
        fetch(drive, "sync", lambda n: n.endswith("image_02/timestamps.txt") or n.endswith("oxts/timestamps.txt"), out)
