import enum
import logging
import matplotlib.pyplot as plt

class DatasetTypes(enum.Enum):
    whole = 0  # Returns all dataset
    small = 1  # Returns only 1000 first rows from dataset

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


def plot_metrics(history):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_f1'], label='Val F1')
    plt.plot(epochs, history['val_precision'], label='Val Precision')
    plt.plot(epochs, history['val_recall'], label='Val Recall')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.tight_layout()
    plt.show()
