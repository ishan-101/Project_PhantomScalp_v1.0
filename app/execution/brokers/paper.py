# app/execution/brokers/paper.py

class PaperBroker:
    def __init__(self):
        self.position = 0

    def submit(self, signal):
        # For now just return a filled status
        return {"status": "filled", "side": signal.direction}
