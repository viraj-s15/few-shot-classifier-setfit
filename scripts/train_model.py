from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from setfit import SetFitModel,SetFitTrainer
from datasets import load_dataset

def model_init(params):
    params = params or {}
    max_iter = params.get("max_iter", 100)
    solver = params.get("solver", "liblinear")
    params = {
        "head_params": {
            "max_iter": max_iter,
            "solver": solver,
        }
    }
    return SetFitModel.from_pretrained("all-mpnet-base-v2", **params)

def find_params(model_init=model_init):
    dataset = load_dataset("Veer15/cancer-text-classification")
    train_ds = dataset["train"]
    test_ds = dataset["test"]
    
    trainer = SetFitTrainer(
        train_dataset=train_ds,
        eval_dataset=test_ds,
        model_init=model_init,
    )
    
    hyperparams = {
        "learning_rate": 8.974179864255142e-5,
        "num_epochs": 1,
        "batch_size": 4,
        "seed": 19,
        "num_iterations": 20,
        "max_iter": 204,
        "solver": "liblinear"
    }
    
    trainer.apply_hyperparameters(hyperparams, final_model=True)
    trainer.train()

    trainer._save_pretrainer("../model/")
    
def main():
    find_params()
    
if __name__ == "__main__":
    main()