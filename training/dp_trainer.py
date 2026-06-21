"""Differentially Private training loop — Opacus + AMP."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
from torch.cuda.amp import autocast, GradScaler
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Callable, Optional
from config import DistillConfig
from training.distill_loss import DistillationLoss


def train_with_dp(
    student,
    teacher,
    dataloader: DataLoader,
    cfg: DistillConfig,
    device: str = "cpu",
    progress_callback: Optional[Callable] = None,
):
    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.learning_rate)
    loss_fn = DistillationLoss(temperature=cfg.temperature, alpha=cfg.alpha)
    use_amp = device.startswith("cuda")
    scaler = GradScaler(enabled=use_amp)

    privacy_engine = PrivacyEngine()
    student, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
        module=student,
        optimizer=optimizer,
        data_loader=dataloader,
        epochs=cfg.num_epochs,
        target_epsilon=cfg.dp_epsilon,
        target_delta=cfg.dp_delta,
        max_grad_norm=cfg.dp_max_grad_norm,
    )

    student.train()
    teacher.eval()

    for epoch in range(1, cfg.num_epochs + 1):
        total_loss = 0.0
        steps = 0

        with BatchMemoryManager(
            data_loader=dataloader,
            max_physical_batch_size=cfg.batch_size,
            optimizer=optimizer,
        ) as mem_loader:
            for batch in tqdm(mem_loader, desc=f"Epoch {epoch}/{cfg.num_epochs}"):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                with autocast(enabled=use_amp):
                    with torch.no_grad():
                        teacher_logits = teacher(input_ids).logits
                    student_logits = student(input_ids)
                    loss = loss_fn(student_logits, teacher_logits, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                steps += 1

        eps = privacy_engine.get_epsilon(cfg.dp_delta)
        avg_loss = total_loss / max(steps, 1)
        print(f"Epoch {epoch} | loss={avg_loss:.4f} | ε={eps:.2f} δ={cfg.dp_delta}")

        if progress_callback:
            progress_callback(epoch, cfg.num_epochs, avg_loss, eps)

    return student, privacy_engine