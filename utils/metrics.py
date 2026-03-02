import pandas as pd

class MetricsLogger:
    def __init__(self, filename):
        self.filename = filename
        self.log = []

    def log_step(self, data):
        self.log.append(data)

    def save(self):
        df = pd.DataFrame(self.log)
        df.to_csv(self.filename, index=False)
