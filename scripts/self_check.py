import os

REQUIRED_FILES = [
    "configs/config.yaml",
    "ppo/model.py",
    "ppo/ppo_agent.py",
    "ppo/replay_buffer.py",
    "curriculum/manager.py",
    "training/trainer.py",
    "evaluation/evaluator.py",
    "run_train.py",
    "run_eval.py",
    "requirements.txt",
    "README.md"
]

def main():
    missing_files = []
    for f in REQUIRED_FILES:
        if not os.path.exists(f):
            missing_files.append(f)

    if missing_files:
        print("Missing required files:")
        for f in missing_files:
            print(f"- {f}")
    else:
        print("All required files are present.")

if __name__ == "__main__":
    main()
