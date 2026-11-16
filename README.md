# Linear Regression: Food Truck Profit

This Jupyter notebook implements univariate linear regression with batch gradient descent. It uses city population as the input feature and food truck profit as the target, following the introductory linear regression exercise from Andrew Ng's Coursera Machine Learning course.

## What the notebook does

- Loads and explores the dataset, including summary statistics and a scatter plot.
- Implements the mean squared error cost function.
- Adds an intercept column and prepares the feature and target matrices.
- Trains the model using batch gradient descent.
- Plots the fitted prediction line against the training data.
- Plots cost over the training iterations to show convergence.

## Dataset

The notebook expects a comma-separated, two-column file at `data/ex1data1.txt`, relative to the notebook's working directory. The columns are:

1. Population (in units of 10,000 people)
2. Profit (in units of $10,000)

The notebook assigns these column names itself; the source file should not include a header row. Ensure the `data` directory and file are present before running the data-loading cell.

## Requirements

- Python 3
- Jupyter Notebook or JupyterLab
- NumPy
- pandas
- Matplotlib

Install the Python packages if needed:

```bash
python -m pip install jupyter numpy pandas matplotlib
```

## Run

1. Place `Linear_Regression.ipynb` and the `data` directory in the same project folder.
2. Confirm `data/ex1data1.txt` is present.
3. Start Jupyter from that folder:

   ```bash
   jupyter notebook
   ```

4. Open `Linear_Regression.ipynb` and run the cells from top to bottom.

The notebook uses a learning rate of `0.01` and runs gradient descent for `1000` iterations.

## Model

The notebook fits a straight-line model:

```text
predicted profit = theta_0 + theta_1 * population
```

It initializes both parameters to zero and updates them iteratively to reduce the cost. The final plots show the training observations with the fitted line and the cost curve across iterations.

## Notes

- The notebook uses the legacy `numpy.matrix` interface. Newer NumPy versions may issue deprecation warnings; these do not change the notebook's intended calculations.
- The dataset is not embedded in the notebook. Obtain the `ex1data1.txt` file separately and preserve the expected path and column order.
