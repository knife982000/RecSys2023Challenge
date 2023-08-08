# RecSys2023Challenge - Isistanitos
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/weighted-multi-level-feature-factorization/collaborative-filtering-on-movielens-100k)](https://paperswithcode.com/sota/collaborative-filtering-on-movielens-100k?p=weighted-multi-level-feature-factorization)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/weighted-multi-level-feature-factorization/click-through-rate-prediction-on-criteo)](https://paperswithcode.com/sota/click-through-rate-prediction-on-criteo?p=weighted-multi-level-feature-factorization)

Isistanitos submission for the RecSys Challenge 2023.

## Environment

To run the code, we provide a reduced `environment.yml` for recreating a similar [Conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to the one used for generating the submission. Optionally, these are the relevant libraries and their versions:

* python=3.10.10
* pytorch=2.0.0
* polars=0.17.2
* matplotlib=3.7.1
* tqdm
* scikit-learn=1.2.2
* numba=0.57.0
* jupyterlab=3.6.3

The file `environment_full.yml` describes the exact used Conda environment. Yet, it is tied to the workstation OS and even has hard-coded the paths where the environment was installed.

## Running the model

Before running the model, the dataset should be uncompressed in the folder `sharechat_recsys2023_data`. Our code expects the folders `train` and `test` to be in `sharechat_recsys2023_data`.

### Prepocessing the dataset

Running the notebook `num_log2/Data_preprocess_simple.ipynb` generates two files, namely `num_log2/train.csv` and `num_log2/test.csv` that are required for training the model and generating the predictions.

### Training-inferring.

Running `num_log2/Prediction_deep_mf_single_emb_rnd_v2.ipynb` trains the model, stores the weights and generates the predictions. The predictions are stored in the file `num_log2/log2_out_deep_mf_single_embds_rnd_v2.csv` following the format of the challenge. The weights are stored in the file `num_log2/predict_deep_mf_single_embds_rnd_v2.pt`.

The `num_log2/predict_deep_mf_single_embds_rnd_v2.pt` provided in the repository is the exact model used to generate the submitted predictions. As it is, the notebook ignores the previously stored model and overrides it. 

## Other Experiments

Code for other experiments presented in the paper is presented in the folders `criteo` and `ml100k`.

## Citing

If you found this repository useful, please consider citing:

```bibtex
@misc{rodriguez2023weighted,
      title={Weighted Multi-Level Feature Factorization for App ads CTR and installation prediction}, 
      author={Juan Manuel Rodriguez and Antonela Tommasel},
      year={2023},
      eprint={2308.02568},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```

## Contact info:

* [Antonela Tommasel](https://tommantonela.github.io) (antonela.tommasel@isistan.unicen.edu.ar)
* [Juan Manuel Rodriguez](https://sites.google.com/site/rodriguezjuanmanuel/home) (juanmanuel.rodriguez@isistan.unicen.edu.ar)