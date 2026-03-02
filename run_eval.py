import yaml
from evaluation.evaluator import Evaluator

if __name__ == "__main__":
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    evaluator = Evaluator(config)
    evaluator.evaluate()
