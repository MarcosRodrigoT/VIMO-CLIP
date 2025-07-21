import os
import os.path as osp
import gc
import h5py
import numpy as np
import pandas as pd
import torch
import decord
from decord import VideoReader, cpu
from PIL import Image
from transformers import CLIPImageProcessor, CLIPModel
from tqdm import tqdm


# ────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2048  # frames processed per forward pass
EMB_DTYPE = np.float32  # dtype stored in the HDF5 file
CUDA_HALF = False  # use fp16 on GPU; halves memory use
# ────────────────────────────────────────────────

# Load CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(DEVICE)
if CUDA_HALF and clip_model.device.type == "cuda":
    clip_model = clip_model.half()

clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

decord.bridge.set_bridge("torch")


def _iter_embeddings(vr, indices):
    """
    Generator that yields batches of (idx_chunk, embedding_chunk).
    Keeps GPU / system RAM bounded by BATCH_SIZE.
    """
    for start in range(0, len(indices), BATCH_SIZE):
        idx_chunk = indices[start : start + BATCH_SIZE]

        # 1. read frames (B, H, W, C) uint8 on CPU
        frames = vr.get_batch(idx_chunk)  # torch.uint8, NHWC
        frames = frames.permute(0, 3, 1, 2)  # NCHW

        # 2. convert to PIL (ClipImageProcessor expects images or tensors)
        pil_imgs = [Image.fromarray(f.permute(1, 2, 0).numpy()) for f in frames]

        # 3. preprocess & move to GPU
        inputs = clip_processor(images=pil_imgs, return_tensors="pt")
        inputs = {k: v.to(DEVICE, dtype=torch.float16 if CUDA_HALF else torch.float32) for k, v in inputs.items()}

        # 4. forward pass
        with torch.inference_mode():
            emb = clip_model.get_image_features(**inputs)  # (B, 512)

        emb = emb.float().cpu().numpy().astype(EMB_DTYPE)

        # 5. free everything early
        del frames, pil_imgs, inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        yield idx_chunk, emb


def create_hdf5_dataset(data_root, annotation_file, class_file, output_hdf5, mode):
    """
    Extract CLIP embeddings for every frame in the split specified by *mode*
    and store them (and the labels) in *output_hdf5*.
    """
    # ── housekeeping ────────────────────────────
    out_dir = osp.dirname(output_hdf5)
    if out_dir and not osp.exists(out_dir):
        os.makedirs(out_dir)

    class_df = pd.read_csv(class_file)
    num_classes = len(class_df)
    class_to_idx = {row["id"]: row["name"] for _, row in class_df.iterrows()}

    ann_df = pd.read_csv(annotation_file, header=None, sep=r"\s+")
    ann_list = ann_df.values.tolist()

    with h5py.File(output_hdf5, "w") as hf:
        # global attrs
        hf.attrs["num_classes"] = num_classes
        hf.attrs["dataset_name"] = "MammalNet"
        hf.attrs["type"] = mode
        hf.attrs["clip_model"] = "ViT-B/32"

        # prepare a string array with video‑ids at the end
        vid_ids = []

        # processing loop
        for video_info in tqdm(ann_list, desc=f"Procesando videos en modo {mode}"):
            video_id = video_info[0]
            labels = [int(l) for l in video_info[1:]]
            video_fp = osp.join(data_root, video_id)

            if not osp.exists(video_fp):
                print(f"Video no encontrado: {video_fp}")
                continue
            else:
                print(f"Procesando video {video_id}")

            try:
                vr = VideoReader(video_fp, ctx=cpu(0))
                frame_count = len(vr)
                indices = np.arange(frame_count)

                # allocate an extendable dataset (T × 512)
                g = hf.create_group(video_id)
                emb_ds = g.create_dataset(
                    "embeddings",
                    shape=(0, 512),
                    maxshape=(None, 512),
                    dtype=EMB_DTYPE,
                    compression="gzip",
                    chunks=(BATCH_SIZE, 512),
                )

                # multi‑hot labels
                mhot = np.zeros(num_classes, dtype=np.float32)
                for lbl in labels:
                    if lbl in class_to_idx:
                        mhot[lbl] = 1.0
                    else:
                        print(f"Advertencia: etiqueta {lbl} no encontrada en class_to_idx")

                g.create_dataset("labels", data=mhot)
                g.attrs["total_frames"] = frame_count
                g.attrs["original_frames"] = frame_count

                # ── frame‑wise embedding extraction ──────────────
                offset = 0
                for _, emb_chunk in _iter_embeddings(vr, indices):
                    new_size = offset + emb_chunk.shape[0]
                    emb_ds.resize(new_size, axis=0)
                    emb_ds[offset:new_size] = emb_chunk
                    offset = new_size

                vid_ids.append(video_id)

            except Exception as e:
                print(f"\nError procesando {video_id}: {e}")
                continue

        # quick‑lookup list of processed ids
        hf.create_dataset(
            "video_ids",
            data=np.array(vid_ids, dtype=h5py.string_dtype()),
        )


# ────────────────────────────────────────────────
# Script entry point
# ────────────────────────────────────────────────
if __name__ == "__main__":
    video_dir = "/mnt/Data/mrt/mammalnet/"
    annotation_dir = "dataset/annotations"
    class_file = "dataset/annotations/mn_action.csv"
    embedding_dir = "/mnt/Data/mrt/mammalnet/embeddings"

    if not osp.exists(embedding_dir):
        os.makedirs(embedding_dir)

    for mode in ["train", "val", "test"]:
        ann_file = f"{annotation_dir}/mn_{mode}.csv"
        out_h5 = f"{embedding_dir}/mn_{mode}_clip_vit32.h5"
        create_hdf5_dataset(video_dir, ann_file, class_file, out_h5, mode)
