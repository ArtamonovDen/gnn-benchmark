from models.vanila_gcn import VanilaGCN


class ModelController:
    __models = {
        "vanila_gcn": VanilaGCN,
    }

    @classmethod
    def get_model(cls, model_name: str, **kwargs):
        return cls.__models[model_name](**kwargs)

    @classmethod
    def get_available_models(cls):
        return list(cls.__models.keys())
