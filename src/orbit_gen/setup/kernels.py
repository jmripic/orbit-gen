from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError
from platformdirs import user_cache_dir


@dataclass(frozen=True)
class KernelSpec:
    filename: str
    url: str


REQUIRED_KERNELS: tuple[KernelSpec, ...] = (
    KernelSpec(
        filename="naif0012.tls",
        url="https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls",
    ),
    KernelSpec(
        filename="gm_de440.tpc",
        url="https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/gm_de440.tpc",
    ),
    KernelSpec(
        filename="pck00011.tpc",
        url="https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00011.tpc",
    ),
)


def cache_dir() -> Path:
    direc = Path(user_cache_dir("orbit-gen")) / "kernels"
    direc.mkdir(parents=True, exist_ok=True)
    return direc


def ensure_kernel(spec: KernelSpec) -> Path:
    cache = cache_dir()
    dest = cache / spec.filename

    if dest.exists() and dest.stat().st_size > 0:
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urlopen(spec.url, timeout=30) as resp, open(tmp, "wb") as f:
            f.write(resp.read())
    except URLError as e:
        raise RuntimeError(
            f"Failed to download {spec.filename} from {spec.url}: {e}"
        ) from e
    tmp.rename(dest)
    return dest


def ensure_kernels() -> dict[str, Path]:
    return {spec.filename: ensure_kernel(spec) for spec in REQUIRED_KERNELS}
