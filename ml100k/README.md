# Experiment for the Movie Lens 100k dataset

The dataset is the available at [ML100k](https://grouplens.org/datasets/movielens/100k/).

## Downloading the dataset

For downloading the dataset, you should run: 

```bash
./download_ml100k.sh
``` 

## Experiments 


| Experiment | RMSE | Experiment |
|------------|:-----:|------------|
| Original | 0.936 | ml100k-exp-full-batch.ipynb |
| Original - Linear | 0.939 | ml100k-exp-full-batch-linear.ipynb |
| Original $\sigma_\lambda = 0.25$ | 0.963 | ml100k-exp-full-batch-n25.ipynb| 
| Original No Noise | 1.102 | ml100k-exp-full-batch-nonoise.ipynb |
| Original + feature user<sub>avg</sub>  + user<sub>%std</sub> + movie<sub>avg</sub> + movie<sub>%std</sub> | 0.928 | ml100k-exp-full+mean+std-batch.ipynb |
| Original - Linear + feature feature user<sub>avg</sub>  + user<sub>%std</sub> + movie<sub>avg</sub> + movie<sub>%std</sub> | 0.928 | ml100k-exp-full+mean+std-batch-linear.ipynb | 

