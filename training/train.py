import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from opacus.validators import ModuleValidator
from config import DistillConfig
from models.teacher import load_teacher
from models.student import load_student
from training.dp_trainer import train_with_dp
from typing import Callable, Optional
import json

class NotesDataset(Dataset):
    def __init__(self, path, tokenizer, max_len):
        self.samples = []
        with open(path) as f:
            for line in f:
                text = json.loads(line)["text"]
                enc = tokenizer(text, truncation=True, max_length=max_len,
                                padding="max_length", return_tensors="pt")
                ids = enc["input_ids"].squeeze()
                self.samples.append({"input_ids": ids, "labels": ids.clone()})

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


def main(progress_callback: Optional[Callable] = None):
    cfg = DistillConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg.output_dir, exist_ok=True)

    teacher, tokenizer = load_teacher(cfg.teacher_model_name, device)
    student = ModuleValidator.fix(load_student(cfg, device))

    dataset = NotesDataset("data/sample_notes.jsonl", tokenizer, cfg.max_seq_len)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

    student, _ = train_with_dp(student, teacher, loader, cfg, device,
                                progress_callback=progress_callback)

    ckpt_path = os.path.join(cfg.output_dir, "student.pt")
    torch.save(student._module.state_dict(), ckpt_path)
    print(f"Checkpoint saved → {ckpt_path}")

if __name__ == "__main__":
    main()