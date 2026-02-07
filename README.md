# Fashion MNIST Classifier

Simple TensorFlow/Keras classifier for the Fashion-MNIST dataset with basic evaluation plots and a small Tkinter GUI to predict a single image.

## Project Structure

- Dataset/
  - data.py: exports the Fashion-MNIST dataset to CSV.
- scr/
  - data_loader.py: loads CSV, splits data, normalizes data.
  - model.py: model definition, training, evaluation, save/load.
  - utils.py: plots, confusion matrix, image preprocessing.
  - main.py: training pipeline and GUI prediction flow.
- model.keras: saved model (generated after training).

## Requirements

- Python 3.9+
- tensorflow
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- pillow

You can install dependencies with:

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn pillow
```

## Dataset

The training code expects a CSV at a hardcoded path in `scr/data_loader.py`:

```
C:\Users\Amr\Desktop\project\Dataset\fashion_mnist.csv
```

Options:
- Update the path in `load_data()` to point to your CSV, or
- Generate the CSV with `Dataset/data.py` and then move it to that path.

To generate the CSV in the current folder:

```bash
python Dataset/data.py
```

## Train and Evaluate

Run the main training pipeline:

```bash
python scr/main.py
```

This will:
- Load the CSV
- Train a simple dense network
- Save the model to `model.keras`
- Evaluate and plot confusion matrix
- Show samples of misclassified and correctly classified images

## Predict a Single Image (GUI)

After training, the GUI prompts for an image path and shows the predicted class.
The image is converted to grayscale and resized to 28x28.

If you only want to run the GUI, you can comment out the training call in `main.py`
or add a separate entry point.

## Notes

- The current model is a small baseline (Flatten -> Dense 100 -> Dense 10).
- The plots in `utils.py` are shown interactively.
- `plot_training()` expects validation history; add validation data if you want that chart.
