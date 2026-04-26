from enum import Enum
from typing import Any
 

class ExtendedEnumMixin(Enum):
    @classmethod
    def keys(cls) -> list[str]:
        return [attr.name for attr in cls]

    @classmethod
    def values(cls) -> list:
        return [attr.value for attr in cls]

    @classmethod
    def items(cls) -> dict[str, Any]:
        return {attr.name: attr.value for attr in cls}
    
    def __str__(self):
        return self.value
    
    def __repr__(self):
        return self.value

 
class AggregationMethods(str, ExtendedEnumMixin):
    clustering = "clustering"                   # k-means clustering of the steering vectors over all examples
    mean = "mean"                               # mean of the steering vectors over all examples
    none = "none"                               # no aggregation, returns a steering vector for each example
    pca = "pca"  
 