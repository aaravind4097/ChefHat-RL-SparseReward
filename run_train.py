import yaml
from training.trainer import Trainer

if __name__ == "__main__":
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    trainer = Trainer(config)
    trainer.train()
