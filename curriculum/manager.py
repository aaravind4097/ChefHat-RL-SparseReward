import yaml
from typing import List, Dict, Any

class CurriculumManager:
    """Manages the curriculum learning stages for the PPO agent."""

    def __init__(self, config_path: str):
        """Initializes the CurriculumManager.

        Args:
            config_path: Path to the YAML configuration file.
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.stages: List[Dict[str, Any]] = config["curriculum_stages"]
        self.current_stage_index = 0

    def get_current_stage(self) -> Dict[str, Any]:
        """Returns the current curriculum stage."""
        return self.stages[self.current_stage_index]

    def get_current_opponents(self) -> List[str]:
        """Returns the list of opponents for the current stage."""
        return self.get_current_stage()["opponents"]

    def check_promotion(self, current_win_rate: float) -> bool:
        """Checks if the agent should be promoted to the next stage.

        Args:
            current_win_rate: The agent's win rate in the current stage.

        Returns:
            True if the agent is promoted, False otherwise.
        """
        if self.is_finished():
            return False

        stage = self.get_current_stage()
        if current_win_rate >= stage["win_rate_threshold"]:
            self.current_stage_index += 1
            print(f"\n--- PROMOTION ---")
            if self.is_finished():
                print("Curriculum complete!")
            else:
                next_stage = self.get_current_stage()
                print(f"Promoted to: {next_stage['name']}")
            print("-----------------\n")
            return True
        return False

    def is_finished(self) -> bool:
        """Returns True if the curriculum is complete."""
        return self.current_stage_index >= len(self.stages)
