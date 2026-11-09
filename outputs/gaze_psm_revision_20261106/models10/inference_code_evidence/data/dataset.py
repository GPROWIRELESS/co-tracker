"""Reusable video I/O, gaze validity, target construction and worker caches."""

from collections import OrderedDict
import zlib
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from .io import ClipItem, load_gaze, normalize_items
from dvss_pytorch.utils.spatial import gaze_heatmap


def video_decoder(path):
    # Cached-feature workflows do not need to import/load FFmpeg or torchcodec.
    from torchcodec.decoders import VideoDecoder

    return VideoDecoder(str(path))


class DVSSDataset(Dataset):
    """Frame/sequence dataset. Temporal subclasses only change window indexing.

    Samples always keep a time axis: images[T,3,H,W], coordinates[T,2],
    validity[T]. Anchor labels use video row indices, not raw Frame_Number.
    RGB uint8 resize before /255 matches the retained FiLM feature caches.
    """

    def __init__(
        self,
        items,
        target_type="coord",
        segment_len=1,
        segment_stride=4,
        frame_step=1,
        image_hw=(256, 256),
        label_hw=(64, 80),
        gaussian_sigma_px=5.0,
        filter_mode="drop",
        active_keep=None,
        passive_keep=None,
        strict_len_match=True,
        decoder_cache_size=2,
        max_segments_per_video=None,
        compose=None,
        data_root=None,
        **kwargs,
    ):
        if min(segment_len, segment_stride, frame_step) < 1 or decoder_cache_size < 1:
            raise ValueError(
                "Sequence lengths, strides and cache capacity must be positive"
            )
        if target_type not in ("coord", "saliency") or filter_mode not in (
            "drop",
            "none",
        ):
            raise ValueError("Unsupported target/filter mode")
        unsupported = set(kwargs) - {
            "return_type",
            "return_valid_mask",
            "normalize_image",
        }
        if unsupported:
            raise TypeError(f"Unsupported dataset settings: {sorted(unsupported)}")
        if kwargs.get("normalize_image", True) is not True:
            raise ValueError("Images must be RGB float in [0,1]")
        self.clips = normalize_items(items, data_root=data_root)
        self.segment_len, self.segment_stride, self.frame_step = (
            segment_len,
            segment_stride,
            frame_step,
        )
        self.image_hw, self.label_hw = tuple(image_hw), tuple(label_hw)
        self.target_type, self.gaussian_sigma_px = target_type, gaussian_sigma_px
        self.active_keep, self.passive_keep = active_keep, passive_keep
        self.filter_mode, self.decoder_cache_size = filter_mode, decoder_cache_size
        self.compose = compose
        self.max_segments_per_video = max_segments_per_video
        self.index = []
        self._lengths = []
        self._decoders = OrderedDict()
        self._gazes = OrderedDict()
        for ci, clip in enumerate(self.clips):
            gaze = load_gaze(clip.gaze_npy_path)
            decoder = video_decoder(clip.video_path)
            length = len(decoder)
            del decoder
            if strict_len_match and length != len(gaze["gaze_x"]):
                raise ValueError(f"Video/gaze length mismatch: {clip.video_path}")
            length = min(length, len(gaze["gaze_x"]))
            self._lengths.append(length)
            valid = self._validity(gaze, clip.domain)
            anchors = self._anchors(length)
            if max_segments_per_video is not None:
                anchors = anchors[:max_segments_per_video]
            for anchor in anchors:
                if (
                    filter_mode == "none"
                    or valid[self._required_indices(anchor, length)].all()
                ):
                    self.index.append((ci, int(anchor)))

    def _anchors(self, length):
        span = (self.segment_len - 1) * self.frame_step
        return np.arange(0, max(0, length - span), self.segment_stride, dtype=np.int64)

    def _frame_indices(self, anchor, length):
        return anchor + np.arange(self.segment_len, dtype=np.int64) * self.frame_step

    def _required_indices(self, anchor, length):
        return self._frame_indices(anchor, length)

    def _validity(self, gaze, domain):
        mask = np.isfinite(gaze["gaze_x"]) & np.isfinite(gaze["gaze_y"])
        keep = self.passive_keep if domain == "passive" else self.active_keep
        return mask if keep is None else mask & np.isin(gaze["valid"], list(keep))

    def _cached(self, cache, path, load):
        if path in cache:
            value = cache.pop(path)
        else:
            value = load(path)
        cache[path] = value
        while len(cache) > self.decoder_cache_size:
            cache.popitem(last=False)
        return value

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_decoders"] = OrderedDict()
        state["_gazes"] = OrderedDict()
        # A compact byte payload avoids serial worker startup on large Python
        # lists and does not depend on platform-specific shared file descriptors.
        index = np.asarray(state.pop("index"), dtype=np.int64).reshape(-1, 2)
        state["_packed_index"] = zlib.compress(index.tobytes())
        return state

    def __setstate__(self, state):
        if "_packed_index" in state:
            index = np.frombuffer(zlib.decompress(state.pop("_packed_index")), dtype=np.int64)
            state["index"] = [tuple(row) for row in index.reshape(-1, 2).tolist()]
        self.__dict__.update(state)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        return self._sample(index)

    def __getitems__(self, indices):
        """Decode the union of overlapping windows once per clip in a batch."""
        groups = {}
        for position, index in enumerate(indices):
            ci, anchor = self.index[index]
            ids = self._frame_indices(anchor, self._lengths[ci])
            groups.setdefault(ci, []).append((position, index, ids))
        samples = [None] * len(indices)
        for ci, entries in groups.items():
            ids = np.unique(np.concatenate([entry[2] for entry in entries]))
            path = self.clips[ci].video_path
            decoder = self._cached(self._decoders, path, video_decoder)
            frames = decoder.get_frames_at(ids.tolist()).data
            # Bound resize intermediates for high-resolution source videos,
            # and resize repeated context frames only once per batch.
            frames = torch.cat([
                TF.resize(part, list(self.image_hw), antialias=True).float() / 255
                for part in frames.split(32)
            ])
            for position, index, window in entries:
                selected = frames[torch.from_numpy(np.searchsorted(ids, window))]
                samples[position] = self._sample(index, selected, resized=True)
        return samples

    def _sample(self, index, images=None, resized=False):
        ci, anchor = self.index[index]
        clip = self.clips[ci]
        ids = self._frame_indices(anchor, self._lengths[ci])
        gaze = self._cached(self._gazes, clip.gaze_npy_path, load_gaze)
        if images is None:
            decoder = self._cached(self._decoders, clip.video_path, video_decoder)
            images = decoder.get_frames_at(ids.tolist()).data
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected RGB video frames[T,3,H,W]")
        if not resized:
            images = TF.resize(images, list(self.image_hw), antialias=True).float() / 255
        if self.compose is not None:
            images = self.compose(images)
        xy = np.stack([gaze["gaze_x"][ids], gaze["gaze_y"][ids]], -1)
        finite = np.isfinite(xy).all(-1)
        coords = torch.from_numpy(np.clip(np.nan_to_num(xy), 0, 1).astype(np.float32))
        valid = self._validity(gaze, clip.domain)[ids]
        sample = dict(
            images=images,
            target_coord_norm=coords,
            is_valid=torch.from_numpy(valid.copy()),
            valid=torch.from_numpy(np.asarray(gaze["valid"][ids], dtype=np.int64)),
            frame_indices=torch.from_numpy(ids),
            clip_id=ci,
            anchor=int(ids[-1]),
            frame_number=torch.from_numpy(
                np.asarray(gaze["Frame_Number"][ids], dtype=np.int64)
            ),
            timestamp_ns=torch.from_numpy(
                np.asarray(gaze["Timestamp(ns)"][ids], dtype=np.int64)
            ),
            video_path=clip.video_path,
            gaze_npy_path=clip.gaze_npy_path,
            domain=clip.domain,
            meta=clip.meta or {},
        )
        if self.target_type == "saliency":
            maps = gaze_heatmap(coords, self.label_hw, self.gaussian_sigma_px)[:, None]
            maps[~torch.from_numpy(valid & finite)] = 0
            sample["target_saliency"] = maps
        return sample
