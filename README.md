# Investigating data-driven surrogates for the Ainslie wake model

**Authors:**
* Niccolò Morabito, email: morabito.niccolo@gmail.com
* Erik Quaeghebeur, email: e.quaeghebeur@tue.nl
* Laurens Bliek, email: l.bliek@tue.nl

This repository contains resources related to the publication of the work about machine learning surrogates for the Ainslie wake model. For detailed information about the work, please refer to the final paper.

## Setup
Install the required dependencies:

```
pip install -r requirements.txt
```

In case the Eddy Viscosity (Ainslie) Model has not been merged yet in the main branch of [PyWake repository](https://topfarm.pages.windenergy.dtu.dk/PyWake/), the following additional steps are required:

```
git clone https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake.git
cd PyWake
git checkout cj_add_eddy_viscosity_model
pip install -e .[test]
echo "\nfrom .eddy_viscosity import EddyViscosityDeficitModel, EddyViscosityModel" >> py_wake/deficit_models/__init__.py
```

## Data
The data utilized for this project was generated using the [PyWake implementation](https://topfarm.pages.windenergy.dtu.dk/PyWake/) of the Eddy Viscosity (Ainslie) model.

Relevant notebooks:
* [`ainslie_data_generation.ipynb`](notebooks/ainslie_data_generation.ipynb) - Data generation process;
* [`data_analysis.ipynb`](notebooks/analysis/data_analysis.ipynb) - Data analysis and exploration.

## Codebase
The helper and utility code are located in the [`src/`](src/) folder.

## Models
Multiple models have been developed in this study to compare their generalization capabilities. The code for the trained models can be found in the notebooks within the [`notebooks/training/`](notebooks/training/) folder. The trained model weights are saved in the [`saved_models/`](saved_models/) folder.

## Results
Different results obtained from experiments are saved in the [`metrics/`](metrics/) folder:
* [`metrics/logged_metrics/`](metrics/logged_metrics/) - Contains logged training information (e.g., training and validation loss values);
* [`metrics/final_results/`](metrics/final_results/) - Includes results from various interpolation and extrapolation experiments. Results are categorized based on whether they are from the train or test set and the type of experiment.

Relevant notebooks:
* [`logged_metrics_visualizations.ipynb`](notebooks/analysis/logged_metrics_visualizations.ipynb) - Visualizations of logged metrics;
* [`result_analysis.ipynb`](notebooks/analysis/result_analysis.ipynb) - Visualization, comparison, and further analysis of experiment results.
