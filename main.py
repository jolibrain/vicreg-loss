from src.trainer import VICRegTrainer

trainer = VICRegTrainer()
trainer.launch_training(1, "cuda")
