# Detecting Human Vs. LLM-Generated Text

## Author: 
- Nikhil Dronamraju
- nikhildronam@arizona.edu

## Requirements:
- Python 3.13+
- Tensorflow
- Keras
- Numpy

## Setup and running:
This project runs using Python's virtual environment. In order to use this, run the command `source venv/bin/activate`. This will create a virtual environment containing all of the relevant dependencies and information. Afterwards, install dependencies with `pip install -r requirements.txt`.

### Data Loading:
On the data, please download the data from [this URL](https://www.kaggle.com/datasets/starblasters8/human-vs-llm-text-corpus). Unzip the file and place all the contents into `/data/raw`. After that, run the preprocessing script with `python3 src/preprocessing.py` in order to receive clean data for the experiment.

In order to train, just running the training script with `python3 src/train.py`

Similarly, executing the `src/evaluate.py` script will provide metrics.

## Results:
- A hybrid CNN-LSTM approach proved to be better than raw CNNs and RNNs for text classification of LLM vs human-generated texts.
