import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        
        # KL Divergence aligns the student's probability distribution with the teacher's
        self.kl = nn.KLDivLoss(reduction="batchmean")
        
        # Cross Entropy aligns the student with the actual ground-truth text
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, student_logits, teacher_logits, labels):
        V = student_logits.shape[-1]
        
        # Apply temperature scaling to soften the probabilities
        soft_student = F.log_softmax(student_logits.view(-1, V) / self.T, dim=-1)
        soft_teacher = F.softmax(teacher_logits.view(-1, V) / self.T, dim=-1)

        kl_loss = self.kl(soft_student, soft_teacher) * (self.T ** 2)
        ce_loss = self.ce(student_logits.view(-1, V), labels.view(-1))
        
        # Blend the losses based on the alpha hyperparameter
        return self.alpha * kl_loss + (1 - self.alpha) * ce_loss