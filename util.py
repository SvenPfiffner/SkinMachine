import json

class HParams:

    def __init__(self, path="hparams.json"):
        self.path = path
        self.data = self.load_hparams()

    def load_hparams(self):
        try:
            with open(self.path, "r") as f:
                data = json.load(f)
            return data
        except:
            return None
        
    def __str__(self):
        if self.data:
            return str(self.data)
        else:
            return "No hyperparameters loaded"
    
    def __repr__(self):
        return str(self)