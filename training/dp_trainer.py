"""
Differentially Private training loop using Opacus and Automatic Mixed Precision (AMP).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
from torch.cuda.amp import autocast, GradScaler
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import DistillConfig
from training.distill_loss import DistillationLoss

def train_with_dp(
    student,
    teacher,
    dataloader: DataLoader,
    cfg: DistillConfig,
    device: str = "cpu",
):
    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.learning_rate)
    loss_fn = DistillationLoss(temperature=cfg.temperature, alpha=cfg.alpha)

    # Initialize the AMP GradScaler. It will automatically disable itself if running on CPU.
    use_amp = device.startswith("cuda")
    scaler = GradScaler(enabled=use_amp)

    # Attach Opacus PrivacyEngine
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
    for epoch in range(cfg.num_epochs):
        total_loss = 0.0
        
        # Opacus BatchMemoryManager handles the logical vs physical batching math
        with BatchMemoryManager(
            data_loader=dataloader,
            max_physical_batch_size=cfg.batch_size,
            optimizer=optimizer,
        ) as mem_dataloader:
            
            for batch in tqdm(mem_dataloader, desc=f"Epoch {epoch+1}"):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()

                # AMP Context Manager: Casts operations to FP16/BF16 where safe
                with autocast(enabled=use_amp):
                    # Teacher forward (no grad)
                    with torch.no_grad():
                        teacher_out = teacher(input_ids)
                        teacher_logits = teacher_out.logits

                    # Student forward
                    student_logits = student(input_ids)
                    loss = loss_fn(student_logits, teacher_logits, labels)

                # Scale the loss and call backward to create scaled gradients
                # Opacus intercepts this step to clip and add DP noise
                scaler.scale(loss).backward()
                
                # Unscale gradients and step the optimizer
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()

        # Calculate and log the privacy budget consumed this epoch
        eps = privacy_engine.get_epsilon(cfg.dp_delta)
        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | ε = {eps:.2f}, δ = {cfg.dp_delta}")

    return student, privacy_engine