"""Small model-download helper.

Four preprocessors used to import ``load_file_from_url`` from ``basicsr``,
which drags in a package that no longer imports at all on torchvision >= 0.17
(it reaches for the removed ``torchvision.transforms.functional_tensor``).
The function itself is twenty lines, so it lives here instead.
"""
import os
import urllib.parse
import urllib.request

DEFAULT_MODEL_DIR = os.path.join("models", "annotators")


def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    """Download ``url`` into ``model_dir`` and return the local path.

    A file that is already there is reused, so repeated node runs do not
    re-download the weights.
    """
    model_dir = model_dir or DEFAULT_MODEL_DIR
    os.makedirs(model_dir, exist_ok=True)

    if not file_name:
        file_name = os.path.basename(urllib.parse.urlparse(url).path) or "model.pth"
    cached = os.path.abspath(os.path.join(model_dir, file_name))
    if os.path.exists(cached):
        return cached

    print(f"Downloading {url}\n     -> {cached}")
    tmp = cached + ".part"
    try:
        with urllib.request.urlopen(url) as response, open(tmp, "wb") as fh:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if progress and total:
                    print(f"\r  {done * 100 // total}%", end="", flush=True)
        if progress and total:
            print()
        os.replace(tmp, cached)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return cached
