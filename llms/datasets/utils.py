import os
import requests
from tqdm import tqdm
import hashlib


def check_integrity(path, md5):
    """Check if the file at path has the expected md5.
    linux: md5sum path
    """
    if md5 is None:
        return True

    if not os.path.isfile(path):
        return False
    md5o = hashlib.md5()
    with open(path, "rb") as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            md5o.update(chunk)

    print(md5o.hexdigest())
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def download_url(url, pathname, md5=None, force=False):
    if os.path.isfile(pathname) and check_integrity(pathname, md5) and not force:
        print("Using downloaded and verified file: " + pathname)
        return

    path = os.path.dirname(pathname)
    if not os.path.exists(path):
        print("Path {} not exits, we are create ..".format(path))
        os.makedirs(path)

    r = requests.get(url, stream=True)
    with open(pathname, "wb") as f:
        for chunk in tqdm(r.iter_content(chunk_size=1024)):
            f.write(chunk)
