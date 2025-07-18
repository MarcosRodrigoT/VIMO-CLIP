import os
import os.path as osp
import h5py
import numpy as np
import pandas as pd
import torch
import decord
from decord import VideoReader, cpu, gpu
from PIL import Image
from transformers import CLIPImageProcessor, CLIPModel
from tqdm import tqdm


# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

decord.bridge.set_bridge("torch")


def create_hdf5_dataset(data_root, annotation_file, class_file, output_hdf5, mode, max_frames=None):
    """
    Crea un archivo HDF5 con todos los embeddings y metadatos.

    Args:
        data_root (str): Directorio raíz de los videos.
        annotation_file (str): Ruta al archivo de anotaciones (formato: video_id label).
        class_file (str): Ruta al archivo de clases (class_id,class_name).
        output_hdf5 (str): Ruta de salida para el archivo HDF5.
        mode (str): Modo del dataset (train, val, test).
        max_frames (int): Máximo número de frames por video (si es None, se procesan todos).
    """
    # Asegurarse de que el directorio de salida exista
    output_dir = osp.dirname(output_hdf5)
    if output_dir and not osp.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Cargar mapeo de clases
    class_df = pd.read_csv(class_file)
    class_to_idx = {row["id"]: row["name"] for _, row in class_df.iterrows()}
    num_classes = len(class_df)

    # 2. Leer anotaciones
    annotations_df = pd.read_csv(annotation_file, header=None, sep=" ")
    annotations = annotations_df.values.tolist()

    # 3. Crear archivo HDF5 de salida
    with h5py.File(output_hdf5, "w") as hf:
        # Agregar metadatos globales
        hf.attrs["num_classes"] = num_classes
        hf.attrs["dataset_name"] = "MammalNet"
        hf.attrs["type"] = mode
        hf.attrs["clip_model"] = "ViT-B/32"

        # Configurar tipos de datos y compresión
        compression = "gzip"

        # 4. Procesar cada video
        for video_info in tqdm(annotations, desc=f"Procesando videos en modo {mode}"):
            video_id = video_info[0]
            labels_str = video_info[1:]
            video_path = osp.join(data_root, video_id)

            if not osp.exists(video_path):
                print(f"Video no encontrado: {video_path}")
                continue

            try:
                # 5. Procesar video
                vr = VideoReader(video_path, ctx=cpu(0))
                total_frames = len(vr)

                # Muestreo de frames
                if (max_frames is None) or (total_frames <= max_frames):
                    indices = np.arange(total_frames)
                else:
                    step = total_frames // max_frames
                    indices = np.arange(0, total_frames, step)[:max_frames]

                # Leer y preprocesar frames en batch
                frames = vr.get_batch(indices).permute(0, 3, 1, 2)  # (T, C, H, W)

                # 6. Procesar con CLIP
                with torch.no_grad():
                    pil_images = [Image.fromarray(np.transpose(frame.numpy(), (1, 2, 0))) for frame in frames]
                    inputs = clip_processor(images=pil_images, return_tensors="pt")
                    inputs = inputs.to(device)
                    pixel_values = inputs["pixel_values"]
                    embeddings = clip_model.get_image_features(pixel_values).cpu().numpy()

                # 7. Convertir etiquetas a multi-hot
                multi_hot = np.zeros(num_classes, dtype=np.float32)
                video_labels = [int(label) for label in labels_str]
                for label in video_labels:
                    if label in class_to_idx:
                        multi_hot[label] = 1.0
                    else:
                        print(f"Advertencia: Etiqueta {label} no encontrada en class_to_idx")

                # 8. Guardar en HDF5
                video_group = hf.create_group(video_id)
                video_group.create_dataset("embeddings", data=embeddings, compression=compression, chunks=(1, embeddings.shape[1]))
                video_group.create_dataset("labels", data=multi_hot)
                video_group.attrs["total_frames"] = len(indices)
                video_group.attrs["original_frames"] = total_frames

            except Exception as e:
                print(f"\nError procesando {video_id}: {str(e)}")
                continue

        # 9. Agregar índices para acceso rápido
        video_ids = np.array([a[0] for a in annotations], dtype=h5py.string_dtype())
        hf.create_dataset("video_ids", data=video_ids)


if __name__ == "__main__":
    # Paths
    video_dir = "/mnt/Data/mrt/mammalnet/trimmed_video/"
    annotation_dir = "dataset/annotations"
    class_file = "dataset/annotations/mn_action.csv"
    # Output directory for embeddings
    embedding_dir = "/mnt/Data/mrt/mammalnet/embeddings"

    if not osp.exists(embedding_dir):
        os.makedirs(embedding_dir)

    for mode in ["train", "val", "test"]:
        annotation_file = f"{annotation_dir}/mn_{mode}.csv"
        output_hdf5 = f"{embedding_dir}/mn_{mode}_clip_vit32.h5"

        create_hdf5_dataset(video_dir, annotation_file, class_file, output_hdf5, mode, max_frames=None)
