from models.gcn import GCN
from models.gcn2 import GCNII


class ModelController:
    __models = {
        "gcn": GCN,
        "gcn2": GCNII,
    }

    @classmethod
    def get_model(cls, model_name: str, **kwargs):
        return cls.__models[model_name](**kwargs)

    @classmethod
    def get_available_models(cls):
        return list(cls.__models.keys())
