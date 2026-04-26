from typing import Any
import logging

logger = logging.getLogger(__name__)


class SteeringConfig:
    def __init__(self, tokenizer=None, **kwargs):
        """ The steering configuration. """
        # Mandatory arguments
        self.tokenizer = tokenizer

        # Pairs construction args
        self.num_fewshot_examples: int = kwargs.get("num_fewshot_examples", 0)
        self.n_contrastive_samples: int = kwargs.get("n_contrastive_samples", None)
        self.prefix: str = kwargs.get("prefix", "Let's think step by step.")
        self.add_question: bool = kwargs.get("add_question", True)
        self.add_answer: bool = kwargs.get("add_answer", False)
        self.add_prefix: bool = kwargs.get("add_prefix", False)
        self.fewshot_delimeter: str = kwargs.get("fewshot_delimeter", "\n\n")
        self.fewshot_only_in_postive: bool = kwargs.get("fewshot_only_in_postive", False)
        self.use_tokenized_version: bool = kwargs.get("use_tokenized_version", False)
        self.encoding_method: str = kwargs.get("encoding_method", "instruct")
        self.extraction_system_prompt: str = kwargs.get("extraction_system_prompt", None)
        self.add_generation_prompt: bool = kwargs.get("add_generation_prompt", True)
        self.sample_selection_method: str = kwargs.get("sample_selection_method", "distance")
        
        # Steering vector extraction args
        self.extraction_method: str = kwargs.get("extraction_method", "last_token")
        self.take_difference: bool = kwargs.get("take_difference", True)
        self.padding_side: str = kwargs.get("padding_side", "left")
        self.add_special_tokens: bool = kwargs.get("add_special_tokens", False)
        self.aggregation_method: str = kwargs.get("aggregation_method", "mean")

        # Applying steering args
        self.append_special_token: bool = kwargs.get("append_special_token", False)
        self.how: Any = kwargs.get("how", "last")

        # Cache steering args
        # self.c_keys: float = kwargs.get("c_keys", None)
        # self.c_values: float = kwargs.get("c_values", None)
        self.img_keys: float = kwargs.get("img_keys", None)
        self.img_values: float = kwargs.get("img_values", None)
        self.txt_keys: float = kwargs.get("txt_keys", None)
        self.txt_values: float = kwargs.get("txt_values", None)
        self.layers_ids_keys = kwargs.get("layers_ids_keys", [1])
        self.layers_ids_values = kwargs.get("layers_ids_values", [1])

        # Clustering args
        self.n_clusters: int = kwargs.get("n_clusters", None)
        self.cluster_layer_id: int = kwargs.get("cluster_layer_id", None)
        self.clustering_method: str = kwargs.get("clustering_method", "kmeans")
        self.cluster_on: str = kwargs.get("cluster_on", None)

        # Functional args
        self.verbose: bool = kwargs.get("verbose", False)

        # Convert layers_ids_values and layers_ids_keys to int if present
        if self.layers_ids_values and isinstance(self.layers_ids_values, str):
            self.layers_ids_values = [int(i) for i in self.layers_ids_values.split(" ")]

        if self.layers_ids_keys and isinstance(self.layers_ids_keys, str):
            self.layers_ids_keys = [int(i) for i in self.layers_ids_keys.split(" ")]

        # Convert how to int
        if self.how not in ["last"]:
            self.how = int(self.how)

        # Convert extraction_method to int
        if self.extraction_method not in ["last_token", "penultimate_token"]:
            self.extraction_method = int(self.extraction_method)

    def __repr__(self):
        s = "SteeringConfig(\n\t"
        for key, value in self.__dict__.items():
            if isinstance(value, str):
                s += f"{key}='{value}',\n\t"
            elif key == "tokenizer":
                s += f"{key}={self.tokenizer.name_or_path if value is not None else 'None'},\n\t"
            else:
                s += f"{key}={value},\n\t"
        s = s[:-1] + ")"
        return s
    
    def __getitem__(self, name: str) -> Any:
        return self.__dict__[name]

    def set_seed(self, seed: int):
        self.seed = seed
