# Experiment for the Criteo dataset

The dataset is the Criteo_x1 from the [BARS](https://openbenchmark.github.io/BARS). Scripts for downloading and formating the dataset are a modification of the [scripts from BARS](https://github.com/openbenchmark/BARS/tree/main/datasets/Criteo/Criteo_x1).

## Downloading the dataset

For downloading and formatting the dataset, you should run: 

```bash
cd dataset
python download_criteo_x1.py
python convert_criteo_x1.py
```

**NOTE:** These scripts required installing pandas, which is not installed by default using the `environment.yml`. 

## Experiments 

For running the experiments, you should run:

```bash
python CriteoExp[version].py
```

| Experiment | AUC | LogLoss | Experiment |
|------------|:---:|:-------:|------------|
| Original | 0.796 | 0.454 | CriteoExp.py | 
| Original with 64 dimensions | 0.799 | 0.451 | CriteoExpLarge.py | 
| Original $\sigma_\lambda = 0.25$ | 0.800 | 0.45 | CriteoExp-25n.py |
| Original with 64 dimensions $\sigma_\lambda = 0.25$ | 0.802 | 0.448 |  CriteoExpLarge-25n.py |
| Original $\sigma_\lambda = 0.10$ | 0.802 | 0.449 | CriteoExp-10n.py |
| Original with 64 dimensions $\sigma_\lambda = 0.10$ | 0.803 | 0.448 | CriteoExpLarge-10n.py |
| Original No Noise | 0.802 | 0.449 |  CriteoExp-nn.py |
| Original with 64 dimensions No Noise | 0.804 | 0.447 |  CriteoExpLarge-nn.py |

Pre-trained models are available at [models](https://drive.google.com/file/d/1UwLOLo6YmqVxzDh1oPu7Hs3V-T8DDeiC/view?usp=sharing)