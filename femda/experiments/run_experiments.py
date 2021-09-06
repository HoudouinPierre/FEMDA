from .labo_test import labo_test

import json, os

def run_experiments():
    def full_path(path):
        return os.path.join(os.path.abspath(os.path.dirname(__file__)), path)

    with open(full_path("paths.config")) as f:
        data = json.load(f)

    labo_test(path_results_simulated_data=data["path_results_simulated_data"],
              path_results_real_data=data["path_results_real_data"],
              path_dataset=data["path_dataset"])

if __name__ == "__main__":
    run_experiments()