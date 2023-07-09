## Installation

To setup all the required dependencies for training and evaluation, please follow the instructions below:

*[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)* **(Recommended)** - Clone the repository and then create and activate a `sim` conda environment using the provided environment definition:

```shell
conda env create sim python=3.10
conda activate sim
```

*[pip](https://pip.pypa.io/en/stable/getting-started/)* - Clone the repository and then use the provided `requirements.txt` to install the dependencies:

```shell
pip install -r requirements.txt
```

## Data preparation

Create and populate a source folder (images to be matched) and a target folder (possible matching images). Any matching images should be identically named for automatic scoring for testing/evaluation. Note that automatic scoring only works for at most one match.

## Train

Run the following script in a command line or in the run.ipynb file provided in order to train the model. Use --help to display all options and descriptions.

```shell
python train.py --weights ./weights/t1/0.pt --source source_path --target target_path --save-dir ./weights/t1 --learning-rate 1e-3 --batch-size 256 --epochs 50
```


## Test/Evaluation

Run the following script in a command line or in the run.ipynb file provided. Use --help to display all options and descriptions.

```shell
python match.py --weights weight_path --topk 15 --source source_folder_path --target target_folder_path --output-folder output_folder_path
```

