from models.vanila_gcn import VanilaGCN


class ModelController:
    models = {
        "gcn": VanilaGCN,
    }

    @classmethod
    def get_model(cls, model_name: str, **kwargs):
        return cls.models[model_name](**kwargs)
