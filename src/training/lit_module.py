"""
LitNanoLogic — El modulo de entrenamiento de PyTorch Lightning.

Este es el "director de orquesta" que coordina TODO el entrenamiento:
- Toma datos del DataLoader
- Los pasa por el modelo
- Calcula el loss
- Ajusta los pesos
- Loguea metricas
- Guarda checkpoints

¿Por que Lightning y no un loop manual?
    Un loop manual requiere ~200 lineas para manejar:
    GPU, multi-GPU, mixed precision, gradient clipping, checkpoints,
    logging, resuming, progress bars, early stopping...
    Lightning maneja TODO eso. Tu solo defines el "que", no el "como".

Tricks implementados:
    1. Schedule-Free AdamW: optimizer que elimina la necesidad de scheduler.
       No mas warmup, cosine decay, ni eleccion de steps.
       Converge igual o mejor que cosine decay. (Facebook Research, 2024)

    2. Gradient Noise Injection: agrega ruido gaussiano a los gradientes.
       Ayuda a escapar minimos locales malos. Con 6K datos el landscape
       es irregular → mas beneficio del ruido. (Neelakantan et al., Google)

    3. EMA (Exponential Moving Average): mantiene una copia "suavizada"
       de los pesos. Los pesos actuales oscilan; el EMA elimina oscilaciones.
       En inferencia se usan los pesos EMA → predicciones mas estables.

    4. Label Smoothing: en vez de target 100% seguro, suavizar a 90%.
       Evita overconfidence. CrossEntropyLoss ya lo soporta con un param.

    5. Gradient Clipping: limitar gradientes por norma global (max=1.0).
       Mantiene la DIRECCION del gradiente pero limita la magnitud.

    6. Mixed Precision: bf16/fp16 para 2x speedup y 2x menos memoria.
       Lightning lo maneja con precision="bf16-mixed" en el Trainer.

    7. Gradient Accumulation: simular batch sizes grandes sin explotar
       la memoria. accumulate_grad_batches=N en el Trainer.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import lightning as L

from src.model.config import NanoLogicConfig
from src.model.transformer import NanoLogicTransformer
from src.tokenizer.tokenizer import NanoLogicTokenizer
from src.training.dataset import create_dataloader


# =====================================================================
# CONFIGURACION DE ENTRENAMIENTO
# =====================================================================
# Separada del config del modelo porque estos params son de ENTRENAMIENTO,
# no de ARQUITECTURA. El modelo no necesita saber el learning rate.


class TrainingConfig:
    """Hiperparametros de entrenamiento.

    Estos NO afectan la arquitectura del modelo, solo como se entrena.
    Se pueden cambiar entre runs sin afectar la compatibilidad del modelo.
    """

    def __init__(
        self,
        # --- Optimizer ---
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        betas: tuple[float, float] = (0.9, 0.95),
        # --- Batching ---
        batch_size: int = 8,
        accumulate_grad_batches: int = 4,
        # batch efectivo = batch_size * accumulate = 32
        # --- Epochs ---
        max_epochs: int = 30,
        # --- Regularizacion ---
        label_smoothing: float = 0.1,
        gradient_clip_norm: float = 1.0,
        # --- Gradient Noise ---
        gradient_noise: bool = True,
        gradient_noise_eta: float = 0.1,
        gradient_noise_gamma: float = 0.55,
        # --- EMA ---
        ema: bool = True,
        ema_decay: float = 0.999,
        # --- Schedule-Free ---
        use_schedule_free: bool = True,
        # --- Packing ---
        use_packing: bool = True,
        # --- Curriculum Learning ---
        # curriculum_schedule: dict de epoch → max_complexity
        # Ejemplo: {0: 0, 5: 1, 15: 2}
        #   Epochs 0-4:  solo Simple
        #   Epochs 5-14: Simple + Intermediate
        #   Epochs 15+:  Todo
        curriculum_schedule: dict[int, int] | None = None,
        # --- Datos ---
        train_path: str = "data/processed/train.jsonl",
        val_path: str = "data/processed/val.jsonl",
        num_workers: int = 0,
    ) -> None:
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.batch_size = batch_size
        self.accumulate_grad_batches = accumulate_grad_batches
        self.max_epochs = max_epochs
        self.label_smoothing = label_smoothing
        self.gradient_clip_norm = gradient_clip_norm
        self.gradient_noise = gradient_noise
        self.gradient_noise_eta = gradient_noise_eta
        self.gradient_noise_gamma = gradient_noise_gamma
        self.ema = ema
        self.ema_decay = ema_decay
        self.use_schedule_free = use_schedule_free
        self.use_packing = use_packing
        self.curriculum_schedule = curriculum_schedule or {0: 0, 5: 1, 15: 2}
        self.train_path = train_path
        self.val_path = val_path
        self.num_workers = num_workers


# =====================================================================
# LIGHTNING MODULE
# =====================================================================


class LitNanoLogic(L.LightningModule):
    """Modulo Lightning para entrenar NanoLogic.

    Flujo por step:
        1. Recibir batch del DataLoader
        2. Forward pass por el modelo
        3. Calcular loss (cross entropy + z-loss + label smoothing)
        4. [Opcional] Agregar gradient noise
        5. Backward pass (gradientes)
        6. [Gradient Accumulation: repetir 1-5 N veces]
        7. Gradient Clipping (max_norm=1.0)
        8. Optimizer step (ajustar pesos)
        9. [EMA: actualizar pesos suavizados]
        10. Loguear metricas

    Args:
        model_config: Configuracion de la arquitectura del modelo.
        train_config: Configuracion del entrenamiento.
        tokenizer: Tokenizer BPE entrenado.
    """

    def __init__(
        self,
        model_config: NanoLogicConfig,
        train_config: TrainingConfig,
        tokenizer: NanoLogicTokenizer,
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.train_config = train_config
        self.tokenizer = tokenizer

        # Guardar hiperparametros en el checkpoint (reproducibilidad)
        self.save_hyperparameters(ignore=["tokenizer"])

        # ===============================================================
        # Crear el modelo
        # ===============================================================
        self.model = NanoLogicTransformer(model_config)

        # ===============================================================
        # EMA: copia suavizada de los pesos
        # ===============================================================
        if train_config.ema:
            # Registrar copias EMA como buffers (no son parametros,
            # no reciben gradientes, pero se guardan en el checkpoint)
            self._ema_weights: dict[str, torch.Tensor] = {}
            self._ema_initialized = False

        # ===============================================================
        # Metricas para tracking
        # ===============================================================
        self._step_losses: list[float] = []

    # =================================================================
    # TRAINING STEP — el corazon del entrenamiento
    # =================================================================

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Un paso de entrenamiento.

        Args:
            batch: Dict con input_ids, targets, y opcionalmente document_mask.
            batch_idx: Indice del batch en este epoch.

        Returns:
            Loss escalar para backward.
        """
        # Forward pass
        mask = batch.get("document_mask", None)
        outputs = self.model(
            input_ids=batch["input_ids"],
            targets=batch["targets"],
            mask=mask,
        )

        loss = outputs["loss"]

        # ----- Label Smoothing -----
        # Ya se aplica DENTRO de cross_entropy cuando especificamos
        # label_smoothing > 0. Pero nosotros calculamos el loss en el modelo.
        # Lo aplicamos aqui recalculando si label_smoothing > 0.
        if self.train_config.label_smoothing > 0:
            logits = outputs["logits"]
            targets = batch["targets"]
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
                label_smoothing=self.train_config.label_smoothing,
            )
            # Re-agregar z-loss si aplica
            if "z_loss" in outputs:
                loss = loss + outputs["z_loss"]

        # ----- Logging -----
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        if "z_loss" in outputs:
            self.log("train/z_loss", outputs["z_loss"], on_step=False, on_epoch=True)

        # Loguear learning rate actual
        opt = self.optimizers()
        if opt is not None:
            current_lr = opt.param_groups[0]["lr"]
            self.log("train/lr", current_lr, on_step=True, on_epoch=False)

        return loss

    # =================================================================
    # GRADIENT NOISE — inyectar ruido despues del backward
    # =================================================================

    def on_after_backward(self) -> None:
        """Hook que se ejecuta DESPUES del backward, ANTES del optimizer step.

        Aqui inyectamos ruido gaussiano a los gradientes.
        El ruido ayuda a escapar minimos locales malos.

        Formula: noise = sqrt(eta / (1 + t)^gamma) * N(0, 1)
        - eta: escala inicial del ruido (default: 0.1)
        - gamma: rate de decaimiento (default: 0.55)
        - t: step actual

        El ruido decae con el tiempo: al inicio explora mucho,
        al final se estabiliza (como simulated annealing).
        """
        if not self.train_config.gradient_noise:
            return

        t = self.global_step
        eta = self.train_config.gradient_noise_eta
        gamma = self.train_config.gradient_noise_gamma

        # Varianza del ruido (decae con el tiempo)
        variance = eta / ((1 + t) ** gamma)
        std = math.sqrt(variance)

        # Agregar ruido a cada gradiente
        for param in self.model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * std
                param.grad.add_(noise)

    # =================================================================
    # EMA — actualizar pesos suavizados despues de cada optimizer step
    # =================================================================

    def on_before_optimizer_step(self, optimizer: Any) -> None:
        """Loguear la norma de los gradientes (util para debug)."""
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=float("inf"),  # no clipear aqui, solo medir
        )
        self.log("train/grad_norm", grad_norm, on_step=True, on_epoch=False)

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Hook despues de cada batch. Actualiza EMA."""
        if not self.train_config.ema:
            return

        decay = self.train_config.ema_decay

        with torch.no_grad():
            if not self._ema_initialized:
                # Primera vez: inicializar EMA con los pesos actuales
                for name, param in self.model.named_parameters():
                    self._ema_weights[name] = param.data.clone()
                self._ema_initialized = True
            else:
                # Actualizar EMA: ema = decay * ema + (1-decay) * current
                for name, param in self.model.named_parameters():
                    self._ema_weights[name].mul_(decay).add_(param.data, alpha=1.0 - decay)

    def swap_to_ema(self) -> dict[str, torch.Tensor]:
        """Intercambia los pesos actuales por los EMA para inferencia.

        Returns:
            Dict con los pesos originales (para restaurar despues).
        """
        if not self.train_config.ema or not self._ema_initialized:
            return {}

        original_weights: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                original_weights[name] = param.data.clone()
                param.data.copy_(self._ema_weights[name])

        return original_weights

    def swap_from_ema(self, original_weights: dict[str, torch.Tensor]) -> None:
        """Restaura los pesos originales despues de usar EMA."""
        if not original_weights:
            return

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data.copy_(original_weights[name])

    # =================================================================
    # VALIDATION STEP
    # =================================================================

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Evaluacion sin gradientes (mas rapido, sin dropout).

        Si tenemos EMA, evaluamos con los pesos EMA (mas estables).
        """
        # Swap a EMA para evaluacion
        original = self.swap_to_ema()

        mask = batch.get("document_mask", None)
        outputs = self.model(
            input_ids=batch["input_ids"],
            targets=batch["targets"],
            mask=mask,
        )

        # Restaurar pesos originales
        self.swap_from_ema(original)

        self.log("val/loss", outputs["loss"], prog_bar=True, on_epoch=True, sync_dist=True)

        # Calcular perplexity: exp(loss)
        # Mide que tan "sorprendido" esta el modelo por los datos.
        # Perplexity 1 = prediccion perfecta, 8000 = random (vocab_size)
        perplexity = torch.exp(outputs["loss"])
        self.log("val/perplexity", perplexity, on_epoch=True, sync_dist=True)

    # =================================================================
    # CONFIGURE OPTIMIZERS
    # =================================================================

    def configure_optimizers(self) -> Any:
        """Configura el optimizer.

        Dos modos:
        1. Schedule-Free AdamW: optimizer que no necesita scheduler.
           Interpola internamente entre dos sequences de pesos.
           Converge igual o mejor que cosine decay.

        2. AdamW + Cosine Schedule: el approach clasico como fallback.
        """
        # ----- Separar parametros: con y sin weight decay -----
        # Weight decay se aplica SOLO a pesos de capas (no a biases,
        # no a norms, no a embeddings). Esto es estandar.
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # No aplicar weight decay a:
            # - Biases (si los hubiera)
            # - Norms (RMSNorm weight, QK-Norm)
            # - Embeddings
            # - Parametros de 1 dimension (scales, lambdas)
            if param.ndim <= 1 or "norm" in name or "embedding" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.train_config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        if self.train_config.use_schedule_free:
            # ----- Schedule-Free AdamW -----
            try:
                from schedulefree import AdamWScheduleFree

                optimizer = AdamWScheduleFree(
                    param_groups,
                    lr=self.train_config.lr,
                    betas=self.train_config.betas,
                    warmup_steps=100,  # warmup interno del optimizer
                )
                return optimizer

            except ImportError:
                # Fallback si schedulefree no esta instalado
                print(
                    "⚠️  schedulefree no instalado. "
                    "Usando AdamW + CosineAnnealing como fallback. "
                    "Instala con: pip install schedulefree"
                )

        # ----- Fallback: AdamW + Cosine Schedule -----
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.train_config.lr,
            betas=self.train_config.betas,
        )

        # Cosine Annealing: lr decae suavemente de lr_max a lr_min
        # T_max = steps totales estimados
        estimated_steps = self.trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=estimated_steps,
            eta_min=self.train_config.lr * 0.1,  # lr minimo = 10% del maximo
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # actualizar cada step, no cada epoch
            },
        }

    # =================================================================
    # DATALOADERS
    # =================================================================

    def train_dataloader(self) -> Any:
        """DataLoader de entrenamiento con curriculum learning."""
        # Curriculum: determinar max_complexity segun el epoch actual
        max_complexity = 2  # default: todo
        current_epoch = self.current_epoch

        for epoch_threshold in sorted(self.train_config.curriculum_schedule.keys(), reverse=True):
            if current_epoch >= epoch_threshold:
                max_complexity = self.train_config.curriculum_schedule[epoch_threshold]
                break

        return create_dataloader(
            jsonl_path=self.train_config.train_path,
            tokenizer=self.tokenizer,
            config=self.model_config,
            batch_size=self.train_config.batch_size,
            shuffle=True,
            num_workers=self.train_config.num_workers,
            max_complexity=max_complexity,
            use_packing=self.train_config.use_packing,
            drop_last=True,
        )

    def val_dataloader(self) -> Any:
        """DataLoader de validacion (siempre usa todos los datos)."""
        return create_dataloader(
            jsonl_path=self.train_config.val_path,
            tokenizer=self.tokenizer,
            config=self.model_config,
            batch_size=self.train_config.batch_size,
            shuffle=False,
            num_workers=self.train_config.num_workers,
            max_complexity=2,  # validar con TODO
            use_packing=self.train_config.use_packing,
            drop_last=False,
        )

    # =================================================================
    # HOOKS DE SCHEDULE-FREE
    # =================================================================
    # Schedule-Free requiere llamar a optimizer.train() y .eval()
    # al cambiar entre entrenamiento y evaluacion.

    def on_train_epoch_start(self) -> None:
        """Poner optimizer en modo train (Schedule-Free)."""
        opt = self.optimizers()
        if opt is not None and hasattr(opt, "train"):
            opt.train()

    def on_validation_epoch_start(self) -> None:
        """Poner optimizer en modo eval (Schedule-Free)."""
        opt = self.optimizers()
        if opt is not None and hasattr(opt, "eval"):
            opt.eval()
