# Overview

The code contained in this repository is complementary to the publication "Machine learning methods to detect money laundering in the Bitcoin blockchain in the presence of label scarcity".
We provide the necessary code base to reproduce our results from section 4.1 (Supervised baseline) and 4.2 (Anomaly detection). We call the Python project reaml - REsearch for AML. 

## Project structure

    ├── README.md          
    ├── data
    │   ├── elliptic                <- Raw dump of the Elliptic data set (https://www.kaggle.com/ellipticco/elliptic-data-set)
    ├── environment.yml             <- The requirements file for reproducing the analysis environment (can be used with conda)
    ├── src                         <- Folder with all the scripts and Python packadges developed for the project
    │   ├── reaml                   <- Main python package for the project
    │   ├── experiments             <- Python scripts to reproduce the experiments 
    │   │   ├── general functions   <- General functions needed to reproduce the experiments on the Elliptic data set
    ├── output                      <- Folder where all output files from the experiments are saved

## Getting Started

Main requirements (For a full list of requirement, check the `enviroment.yaml` file):

* Python 3.7
* conda
* jupyterlab
* networkx
* sklearn
* matplotlib
* seaborn
* pyod

To setup and activate the environment, run the following commands in the terminal from the project root folder:
```bash
conda env create -f environment.yml
conda activate reaml
```

Verify that the new environment was created successfully:
``` bash
conda env list
```

Additionally, download the Elliptic Bitcoin dataset from https://www.kaggle.com/ellipticco/elliptic-data-set and save the three .csv files in data/elliptic/dataset/.

## Running tests
All Python scripts to reproduce the results are in src/experiments. To reproduce a specific experiment, run the respective python script in the terminal from the project root folder:
- Supervised methods illicit F1-score across time (Figure 2):
```supervised_baseline.py```
- Anomaly detection methods and supervised baseline illicit F1-score by contamination level (Table 1):
```anomaly_detection_benchmark.py```
- UMAP projection of the test set, colored by the predicted labels (Figure 3) and colored by the true labels (Figure 4)
```umap_data_exploration.py```

The results will be saved in the output folder, as either a .png or a .csv file.

## Authors
* Joana Lorenz,       joana.lorenz@outlook.de
* Maria Inês Silva,   maria.silva@feedzai.com
* David Aparício,     david.aparicio@feedzai.com


## License
This work is licensed under the Apache License, Version 2.0 (the "License"). A copy of the License can be found in the project repository and at http://www.apache.org/licenses/LICENSE-2.0. You may not use the content of this work except in compliance with the License.

