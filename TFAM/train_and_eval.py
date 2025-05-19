import argparse
import json
import logging
import os
import time
import yaml
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tabulate import tabulate
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MultilabelAveragePrecision
from tqdm import tqdm
from data import HDF5VideoDataset, collate_fn_pad
from models import AMO_CLIP
from datetime import datetime


# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.FileHandler("training.log"), logging.StreamHandler()])


def set_seed(seed: int = 0):
    """
    Fixes random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may slow down training):
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Inicializar la métrica
        self.mAP_metric = MultilabelAveragePrecision(num_labels=self.config.num_classes, average="micro").to(self.config.device)
        self.best_val_mAP = 0.0  # Track the highest val_mAP

        # Inicializar optimizador y scheduler
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.epochs, eta_min=1e-6  # Ciclo completo sobre todas las épocas  # Learning rate mínimo (opcional, por defecto es 0)
        )

        self.criterion = nn.BCEWithLogitsLoss()
        self.writer = SummaryWriter(self.config.log_dir)
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0

        # Crear directorios
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        self.mAP_metric.reset()
        total_loss = 0.0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, batch in enumerate(progress_bar):
            video_input = batch["embeddings"].to(self.config.device)
            flow_input = batch["flow_embeddings"].to(self.config.device)
            labels = batch["labels"].to(self.config.device)
            mask_rgb = batch["mask_rgb"].to(self.config.device)  # (B, T_rgb_max)
            mask_flow = batch["mask_flow"].to(self.config.device)  # (B, T_flow_max)

            self.optimizer.zero_grad()
            output = self.model(video_input, flow_input, mask_rgb=mask_rgb, mask_flow=mask_flow)
            loss = self.criterion(output, labels)
            loss.backward()
            self.optimizer.step()

            # Acumular métricas (mAP)
            total_loss += loss.item()
            self.mAP_metric.update(output, labels.to(dtype=torch.int))

            # Logging de batch
            if batch_idx % 50 == 0:
                mAP = self.mAP_metric.compute()  # Obtener mAP acumulado hasta ahora
                progress_bar.set_postfix({"loss": loss.item(), "mAP": f"{mAP:.4f}"})

        epoch_mAP = self.mAP_metric.compute()
        train_loss = total_loss / len(self.train_loader)

        # Loggear a TensorBoard
        self.writer.add_scalar("Loss/train", train_loss, epoch)
        self.writer.add_scalar("mAP/train", epoch_mAP, epoch)

        return train_loss, epoch_mAP

    def validate(self, epoch):
        self.model.eval()
        self.mAP_metric.reset()
        total_loss = 0.0

        progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}")
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                video_input = batch["embeddings"].to(self.config.device)
                flow_input = batch["flow_embeddings"].to(self.config.device)
                labels = batch["labels"].to(self.config.device)
                mask_rgb = batch["mask_rgb"].to(self.config.device)  # (B, T_rgb_max)
                mask_flow = batch["mask_flow"].to(self.config.device)  # (B, T_flow_max)

                output = self.model(video_input, flow_input, mask_rgb=mask_rgb, mask_flow=mask_flow)
                loss = self.criterion(output, labels)
                total_loss += loss.item()

                # Calcular batch mAP
                self.mAP_metric.update(output, labels.to(dtype=torch.int))

        val_mAP = self.mAP_metric.compute()
        val_loss = total_loss / len(self.val_loader)

        # Loggear a TensorBoard
        self.writer.add_scalar("Loss/val", val_loss, epoch)
        self.writer.add_scalar("mAP/val", val_mAP, epoch)

        return val_loss, val_mAP

    def save_checkpoint(self, val_loss, val_mAP, epoch, best=False):
        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),  # Guardar estado del scheduler
            "best_val_loss": self.best_val_loss,  # keep if you want
            "best_val_mAP": self.best_val_mAP,
        }
        # Verificar si este es el mejor modelo
        if val_mAP > self.best_val_mAP:
            self.best_val_mAP = val_mAP
            best_filename = "best_model.pth"
            best_save_path = os.path.join(self.config.checkpoint_dir, best_filename)
            torch.save(state, best_save_path)
            logging.info(f"New best model in epoch {epoch} (mAP={val_mAP:.4f}) saved to {best_save_path}")

    def train(self):
        start_time = time.time()

        for epoch in range(self.config.epochs):
            logging.info(f"\nEpoch {epoch+1}/{self.config.epochs}")
            train_loss, train_mAP = self.train_epoch(epoch)
            val_loss, val_mAP = self.validate(epoch)

            # Save checkpoint each epoch
            self.save_checkpoint(val_loss, val_mAP, epoch)

            # Actualizar el learning rate después de cada época
            self.scheduler.step()

            # Registrar el learning rate actual
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("Learning Rate", current_lr, epoch)

            # Loggear métricas
            logging.info(f"Train Loss: {train_loss:.4f} | Train mAP: {train_mAP:.4f} | Val Loss: {val_loss:.4f} | Val mAP: {val_mAP:.4f} | LR: {current_lr:.2e}")

        self.writer.close()
        logging.info(f"Entrenamiento completo en {(time.time() - start_time)/60:.2f} minutos")


class ModelTester:
    def __init__(self, model, val_loader, config):
        self.model = model.to(config.device)
        self.val_loader = val_loader
        self.config = config
        self.mAP_metric = MultilabelAveragePrecision(num_labels=self.config.num_classes, average="micro").to(self.config.device)
        self.criterion = nn.BCEWithLogitsLoss()

        df = pd.read_csv(self.config.class_names_dir, header=None, names=["id", "name"])
        self.class_names = {str(row["id"]): row["name"] for _, row in df.iterrows()}

    def load_best_model(self, checkpoint_dir):
        model_path = os.path.join(checkpoint_dir, "best_model.pth")
        checkpoint = torch.load(model_path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        logging.info(f"Mejor modelo cargado desde {model_path}")

    def evaluate(self, save_predictions=False, top_k=5):
        self.mAP_metric.reset()
        total_loss = 0.0
        results = {"videos": [], "metrics": {}, "config": {"model": self.model.__class__.__name__, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}}

        all_preds = []
        all_labels = []
        all_ids = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluando"):
                video_input = batch["embeddings"].to(self.config.device)
                flow_input = batch["flow_embeddings"].to(self.config.device)
                labels = batch["labels"].to(self.config.device)
                video_ids = batch["video_id"]
                mask_rgb = batch["mask_rgb"].to(self.config.device)  # (B, T_rgb_max)
                mask_flow = batch["mask_flow"].to(self.config.device)  # (B, T_flow_max)

                # modo de inferencia
                outputs = self.model(video_input, flow_input, mask_rgb=mask_rgb, mask_flow=mask_flow)

                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                self.mAP_metric.update(outputs, labels.to(dtype=torch.int))
                probs = torch.sigmoid(outputs).cpu()
                all_preds.append(probs)
                all_labels.append(labels.cpu())
                all_ids.extend(video_ids)

                for i in range(outputs.size(0)):
                    video_result = {"video_id": batch["video_id"][i], "true_labels": [], "predictions": {}}

                    # Obtener probabilidades y clases
                    probs_i = torch.sigmoid(outputs[i]).cpu().numpy()
                    sorted_indices = np.argsort(probs_i)[::-1]

                    # Almacenar top-k predicciones
                    for idx in sorted_indices[:top_k]:
                        video_result["predictions"][str(idx)] = {"class_name": self.class_names.get(str(idx), f"class_{idx}"), "probability": round(float(probs_i[idx]), 4)}

                    # Almacenar etiquetas verdaderas
                    true_labels = np.where(labels[i].cpu().numpy() == 1)[0]
                    for lbl in true_labels:
                        video_result["true_labels"].append({"class_id": str(lbl), "class_name": self.class_names.get(str(lbl), f"class_{lbl}")})

                    results["videos"].append(video_result)

        results["metrics"]["loss"] = total_loss / len(self.val_loader)
        results["metrics"]["mAP"] = self.mAP_metric.compute().item()

        if save_predictions:
            self._save_results(results)

        self._print_terminal_summary(results, top_k)
        return results

    def _save_results(self, results):
        os.makedirs("results", exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"results/results_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        logging.info(f"Resultados guardados en {filename}")

    def _print_terminal_summary(self, results, top_k):
        # Resumen general
        print("\n" + "=" * 60)
        print(f"Resumen de Evaluación ({results['config']['timestamp']})")
        print("=" * 60)
        print(f"Modelo: {results['config']['model']}")
        print(f"\nMétricas Globales:")
        print(f"- Pérdida: {results['metrics']['loss']:.4f}")
        print(f"- mAP: {results['metrics']['mAP']:.4f}")
        print(f"- Videos evaluados: {len(results['videos'])}")

        # Ejemplos de predicciones
        print("\n" + "-" * 60)
        print("Ejemplo de Predicciones (primeros 3 videos):")
        print("-" * 60)

        for video in results["videos"][:3]:
            table_data = []
            print(f"\nVideo ID: {video['video_id']}")

            # Predicciones top-k
            print("\nPredicciones:")
            for cls_id, pred in video["predictions"].items():
                table_data.append([pred["class_name"], f"{pred['probability']:.4f}", "Yes" if any(lbl["class_id"] == cls_id for lbl in video["true_labels"]) else "No"])

            print(tabulate(table_data, headers=["Class", "Probability", "Correct"], tablefmt="pretty"))

            # Etiquetas verdaderas
            true_classes = ", ".join([lbl["class_name"] for lbl in video["true_labels"]])
            print(f"\nEtiquetas Verdaderas: {true_classes}")
            print("-" * 40)

    def run_full_evaluation(self):
        """Ejecuta evaluación y genera report."""
        results = super().run_full_evaluation()

        # Reporte comparativo
        print("\n" + "=" * 60)
        print("Comparativa de Modos de Inferencia")
        print("=" * 60)
        print(
            tabulate(
                [
                    ["Single-view", results["sv"]["metrics"]["loss"], results["sv"]["metrics"]["mAP"]],
                ],
                headers=["Modo", "Pérdida", "mAP"],
                tablefmt="grid",
            )
        )


class Config:
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar y/o evaluar modelo de video")
    parser.add_argument("--config", type=str, default="config_default.yaml", help="Ruta al archivo de configuración (YAML)")
    args = parser.parse_args()

    # Load configuration from config.yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # For clarity, pull out each major section from the dictionary
    train_cfg = cfg["training"]
    log_cfg = cfg["logging"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    config = Config()

    # Train config
    config.mode = train_cfg["mode"]
    config.seed = train_cfg["seed"]
    set_seed(config.seed)
    config.lr = train_cfg["lr"]
    config.epochs = train_cfg["epochs"]
    config.batch_size = train_cfg["batch_size"]
    config.num_workers = train_cfg["num_workers"]
    config.device_str = train_cfg["device"]
    config.device = torch.device(config.device_str if torch.cuda.is_available() else "cpu")

    # Log config
    config.log_dir = log_cfg["log_dir"]
    config.checkpoint_dir = log_cfg["checkpoint_dir"]

    # Data config
    config.num_classes = data_cfg["num_classes"]
    config.class_names_dir = data_cfg["class_names_dir"]
    config.train_dataset_path = data_cfg["train_dataset_path"]
    config.val_dataset_path = data_cfg["val_dataset_path"]
    config.flow_dataset_path = data_cfg["flow_dataset_path"]

    # Model config
    config.d_model = model_cfg["d_model"]
    config.nhead = model_cfg["nhead"]
    config.num_layers = model_cfg["num_layers"]
    config.dim_feedforward = model_cfg["dim_feedforward"]
    config.use_cross_attn = model_cfg["use_cross_attention"]
    config.concat_dim = model_cfg["concat_dim"]
    config.dropout = model_cfg["dropout"]
    config.mlp_dropout = model_cfg["mlp_dropout"]
    config.use_pe = model_cfg["use_pe"]
    config.use_only_rgb = model_cfg["use_only_rgb"]
    config.use_only_flow = model_cfg["use_only_flow"]

    # === Logging & checkpoint setup with a timestamp-based subfolder ===
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")  # e.g., "20231105-183122"
    config.log_dir = os.path.join(args.config.split(".yaml")[0], config.log_dir, run_name)
    config.checkpoint_dir = os.path.join(args.config.split(".yaml")[0], config.checkpoint_dir, run_name)

    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    val_dataset = HDF5VideoDataset(config.val_dataset_path, config.flow_dataset_path, num_frames=None, max_frames=None)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, collate_fn=collate_fn_pad, num_workers=config.num_workers, drop_last=True)

    # Initialize model
    model = AMO_CLIP(
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        num_classes=config.num_classes,
        use_only_rgb=config.use_only_rgb,
        use_only_flow=config.use_only_flow,
        use_pe=config.use_pe,
        use_cross_attention=config.use_cross_attn,
        concat_dim=config.concat_dim,
        dropout=config.dropout,
        mlp_dropout=config.mlp_dropout,
        device=config.device,
    )
    model = nn.DataParallel(model)

    # Ejecutar según modo seleccionado
    if config.mode in ["train", "both"]:
        train_dataset = HDF5VideoDataset(config.train_dataset_path, config.flow_dataset_path, num_frames=None, max_frames=None)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn_pad, num_workers=config.num_workers, drop_last=True)

        trainer = ModelTrainer(model, train_loader, val_loader, config)
        trainer.train()

    if config.mode in ["test", "both"]:
        tester = ModelTester(model, val_loader, config)
        tester.load_best_model(config.checkpoint_dir)
        tester.evaluate(save_predictions=True, top_k=5)
