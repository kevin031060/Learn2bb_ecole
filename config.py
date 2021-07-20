import torch
from pathlib import Path


class Config:
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __init__(self, run_flag = "0", problem = "setcover"):
        self.run_flag = run_flag
        self.problem = problem
        self.log_path = f"log/{self.problem}/{self.run_flag}"
        self.save_path = f"checkpoints/{problem}/{self.run_flag}"
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        Path(self.log_path).mkdir(parents=True, exist_ok=True)
