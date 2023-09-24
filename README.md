# Data-driven Surrogate Models for Predicting Wind Turbine Wake Effects
**Master Thesis for Big Data Management and Analytics**

**Author:** Niccolò Morabito  
**Supervisors:** Erik Quaeghebeur & Laurens Bliek

This repository contains resources related to my thesis research on the development and analysis of Ainslie surrogate models.

For detailed information about the work, please refer to the [final report](<Master Thesis - Report.pdf>).

## Setup
Install the required dependecies:

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
The data utilized for this project was generated using the [PyWake implementation](https://topfarm.pages.windenergy.dtu.dk/PyWake/) of the Eddy Viscosity (Ainslie) model [[1]](#1).

Relevant notebooks:
* [`data_generation_ainslie.ipynb`](data_generation_ainslie.ipynb) - Data generation process;
* [`data_analysis.ipynb`](data_analysis.ipynb) - Data analysis and exploration.

## Models
Multiple models have been developed in this study to compare their generalization capabilities. The code for the trained models can be found in the notebooks within the `learning/` folder. The trained model weights are saved in the `saved_models/` folder.

## Results
Different results obtained from experiments are saved in the `metrics/` folder:
* `metrics/logged_metrics/` - Contains logged training information (e.g., training and validation loss values)
* `metrics/final_results/` - Includes results from various interpolation and extrapolation experiments. Results are categorized based on whether they are from the train or test set and the type of experiment.

Relevant notebooks:
* [`logged_metrics_analysis.ipynb`](logged_metrics_analysis.ipynb) - Analysis of logged metrics
* [`result_analysis.ipynb`](result_analysis.ipynb) - Visualization, comparison, and further analysis of experiment results

## References
<a id="1">[1]</a> John F. Ainslie, "Calculating the flowfield in the wake of wind turbines" (1988). Journal of Wind Engineering and Industrial Aerodynamics.