

class TUDatasetWrapper:

    __supported_types = ["NCI1", "ENZYMES", "DD", "MUTAG", "PROTEINS"]

    # added manually after analysis
    __degrees = {
        "NCI1": 4,
        "ENZYMES": 9,
        "DD": 19,
        "MUTAG": 4,
        "PROTEINS": 25,
    }

    # added manually after analysis
    __diameters = {
        "NCI1": 0, # TODO
        "ENZYMES": 37,
        "DD": 83,
        "MUTAG": 15,
        "PROTEINS": 64,
    }

    @classmethod
    def is_supported(cls, type):
        return type in cls.__supported_types

    @classmethod
    def get_max_degree(cls, type):
        if type in cls.__degrees:
            return cls.__degrees[type]
        raise ValueError("Unsupported dataset type")

    @classmethod
    def get_max_diameter(cls, type):
        if type in cls.__diameters:
            return cls.__diameters[type]
        raise ValueError("Unsupported dataset type")
