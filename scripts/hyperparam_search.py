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


def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "num_epochs": trial.suggest_int("num_epochs", 1,1),
        "batch_size": trial.suggest_categorical("batch_size", [4, 8, ]),
        "seed": trial.suggest_int("seed", 1, 40),
        "num_iterations": trial.suggest_categorical("num_iterations", [5, 10, 20]),
        "max_iter": trial.suggest_int("max_iter", 50, 300),
        "solver": trial.suggest_categorical("solver", ["newton-cg", "lbfgs", "liblinear"]),
    }


def find_params(model_init=model_init,hp_space=hp_space):
    dataset = load_dataset("Veer15/cancer-text-classification")
    train_ds = dataset["train"]
    test_ds = dataset["test"]
    
    trainer = SetFitTrainer(
        train_dataset=train_ds,
        eval_dataset=test_ds,
        model_init=model_init,
    )
    best_run = trainer.hyperparameter_search(direction="maximize", hp_space=hp_space, n_trials=10)
    print("The best hyperparams are:")
    print(best_run.hyperparameters)
    
def main():
    find_params()
    
if __name__ == "__main__":
    main()