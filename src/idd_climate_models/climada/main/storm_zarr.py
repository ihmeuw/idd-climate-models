"""
Open stage 1 storm groups without asking the filesystem about empty chunks.

Stage 1 writes each storm's arrays in 64x64 chunks with a NaN fill value, and
zarr does not write chunks that are entirely NaN (ocean after the land clip).
A plain read then looks up every chunk position, and on a typical storm half
to two-thirds of those lookups miss. Across stages 2-4 that was millions of
NOENT lookups on the shared filesystem.

`ChunkManifestStore` lists each array's chunk directory once (a few readdir
calls) and answers missing chunks from that listing, so absent chunks are
never looked up. Data read through it is identical to a plain read.
"""

import os
from pathlib import Path

import xarray as xr  # type: ignore
from zarr.storage import LocalStore  # type: ignore


class ChunkManifestStore(LocalStore):
    """LocalStore that skips filesystem lookups for chunks that were never written.

    Chunk keys use zarr's default encoding, "<array>/c/<i>/<j>". The first read
    of an array lists "<array>/c/" and every later chunk read is checked
    against that listing. The store is read-only and caches the listing per
    instance, so open a new one per storm (see `open_storm`).
    """

    def __init__(self, root: Path | str, *, read_only: bool = True) -> None:
        super().__init__(root, read_only=read_only)
        self._chunk_listing: dict[str, set[str]] = {}

    def _existing_chunks(self, array: str) -> set[str]:
        listing = self._chunk_listing.get(array)
        if listing is None:
            listing = set()
            chunk_dir = Path(self.root) / array / "c"
            for dirpath, _, filenames in os.walk(chunk_dir):  # no error if the dir is missing
                rel = Path(dirpath).relative_to(self.root).as_posix()
                listing.update(f"{rel}/{name}" for name in filenames)
            self._chunk_listing[array] = listing
        return listing

    async def get(self, key, prototype=None, byte_range=None):
        array, is_chunk, _ = key.partition("/c/")
        if is_chunk and key not in self._existing_chunks(array):
            return None  # never written: zarr fills it with the fill value
        return await super().get(key, prototype, byte_range)


def open_storm(storm_path: Path | str, **kwargs) -> xr.Dataset:
    """xr.open_zarr for one stage 1 storm group (Zarr format 3, unconsolidated).

    Extra keyword arguments (chunks, decode_times, ...) go to xr.open_zarr.
    """
    return xr.open_zarr(
        ChunkManifestStore(storm_path), consolidated=False, zarr_format=3, **kwargs
    )
