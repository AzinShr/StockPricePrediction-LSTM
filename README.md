# Stock Price Prediction with a Stacked LSTM

A deep learning project that predicts the next-day closing price of a stock using a two-layer LSTM network built with TensorFlow/Keras. The model was trained and evaluated on **IBM** stock data, and the same pipeline can be applied to any other company in the dataset by changing a single variable.

## Results (IBM)

| Metric | Value |
|--------|-------|
| MSE    | 15.79 |
| RMSE   | 3.97  |

The RMSE is in the same units as the target (price), so on average the model's predictions are off by about 3.97 units from the true closing price.

> **Tip:** add your naive baseline (predict tomorrow = today) here for context, e.g. `Baseline RMSE: X.XX`. A model is only meaningfully good if it beats this.

## Model Architecture

```python
model = keras.models.Sequential()
model.add(keras.layers.Input(shape=(x_train.shape[1], 1)))
model.add(keras.layers.LSTM(units=128, return_sequences=True))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.LSTM(units=64))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(32, activation="relu"))
model.add(keras.layers.Dense(1))
```

| Layer | Details |
|-------|---------|
| LSTM | 128 units, returns full sequence |
| Dropout | 0.2 |
| LSTM | 64 units |
| Dropout | 0.2 |
| Dense | 32 units, ReLU |
| Dense | 1 unit (predicted price) |

**Training setup**
- Optimizer: Adam (learning rate 1e-3)
- Loss: Mean Squared Error
- Batch size: 32
- Max epochs: 100 with early stopping (patience 10, best weights restored)
- Validation split: 10%

## Dataset

- Historical daily stock prices for multiple companies.
- This project uses **IBM** as the example. To train on another company, filter the dataset by its ticker/name (see [Using a Different Company](#using-a-different-company)).
- Input: a sliding window of past closing prices (single feature).
- Target: the next closing price.

*(Add the dataset source/link here, e.g. Kaggle or Yahoo Finance.)*

## Project Structure

```
.
├── README.md
├── requirements.txt
├── notebook.ipynb        # training and evaluation (rename to match your file)
└── data/                 # dataset (not included, see Dataset section)
```

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

Suggested `requirements.txt`:

```
tensorflow
numpy
pandas
scikit-learn
matplotlib
```

## Usage

1. Place the dataset in the `data/` folder.
2. Open the notebook and run all cells:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. The notebook loads the data, scales it, builds the sliding windows, trains the LSTM, and reports MSE and RMSE on the test set.

## Using a Different Company

Change the company filter at the top of the notebook, for example:

```python
company = "IBM"          # change to any other company in the dataset
df = df[df["Name"] == company]
```

Then rerun all cells. Results will differ per company, so retrain rather than reusing weights from another stock.

## Evaluation

The model is evaluated on a held-out test set using:
- **MSE** (mean squared error)
- **RMSE** (root mean squared error), reported in original price units after inverse scaling

## Limitations

- Uses only past closing prices; other features (volume, open/high/low, technical indicators) are not included.
- Stock prices are noisy and influenced by news and market events that the model cannot see.
- Results are for educational purposes and **should not be used as financial advice**.

