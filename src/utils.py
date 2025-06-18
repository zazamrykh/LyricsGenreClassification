import enum
import logging

class DatasetTypes(enum.Enum):
    whole = 0  # Returns all dataset
    small = 1  # Returns only 1000 first rows from dataset
    hundred = 2 # 100 samples in each dataset
    eight = 3 # 8 samples in each dataset

class Params:
    def __init__(self, exp_name='genre_classification', random_seed=1337, n_epoch=10, batch_size=8, dataset_type=DatasetTypes.whole, 
                 learning_rate=1e-4, weight_decay=1e-5):
        self.random_seed = random_seed
        self.exp_name = exp_name
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.dataset_type = dataset_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
    def __str__(self):
        return ", ".join(f"{k}: {v}" for k, v in vars(self).items())
    
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
