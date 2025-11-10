# cs6847_assignment3
The objective of the assignment is to understand Tensorflow and use it to create a model for New York Taxi dataset cab price prediction.

## **1. Project Overview**

This project implements a **neural network from scratch in TensorFlow (low-level API)** to predict taxi fares in New York City.
The work demonstrates understanding of **TensorFlow as a data flow graph engine** by manually defining variables, operations, and training loops without using high-level frameworks such as **Keras** or **PyTorch**.

The dataset used is the **New York City Taxi Fare Prediction** dataset (Kaggle).
The model predicts `fare_amount` based on the following six input features:

1. `pickup_datetime`
2. `pickup_longitude`
3. `pickup_latitude`
4. `dropoff_longitude`
5. `dropoff_latitude`
6. `passenger_count`

---

## **2. File Structure**

```
Roll_number.zip
│
├── nytaxi_tf_assignment_final.ipynb     # Main notebook (TensorFlow implementation)
├── README.md                            # This file
├── /plots                               # Folder containing loss curves, time comparison plots
├── /logs                                # TensorBoard logs for CPU/GPU/TPU runs
├── /tfrecords                           # TFRecord files generated from CSV
│   ├── train.tfrecord
│   └── test.tfrecord
└── report.pdf / report.docx             # Full written report
```

---

## **3. Requirements**

### **Software**

* Python ≥ 3.8
* TensorFlow ≥ 2.12
* NumPy, Pandas, Matplotlib
* Google Colab or local environment with TensorFlow device runtime
* TensorBoard (for visualization)

### **Installation**

If running locally:

```bash
pip install tensorflow numpy pandas matplotlib
```

If running in **Google Colab**, most packages are pre-installed.
You only need to mount your Google Drive and set the dataset path.

---

## **4. Dataset Setup**

Download the **train.csv** and **test.csv** files from Kaggle:

**Dataset link:**
[https://www.kaggle.com/c/new-york-city-taxi-fare-prediction](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction)

Create a folder in your Google Drive:

```
/MyDrive/NYTaxi/
```

and place:

```
train.csv
test.csv
```

inside it.

In the notebook, update the following path if needed:

```python
DATASET_DIR = '/content/drive/MyDrive/NYTaxi'
```

---

## **5. How to Run the Notebook**

### **Step 1: Open in Google Colab**

* Upload the notebook `cs6847_assignment3.ipynb` to Google Colab.
* Mount your Google Drive:

  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```

### **Step 2: Run Preprocessing**

* The notebook automatically reads `train.csv` and `test.csv`.
* It filters and normalizes features.
* TFRecord files are created for efficient streaming.

### **Step 3: Train the Model**

* The TensorFlow graph is built using placeholders, variables, and ops.
* Model trains for a fixed number of epochs (default 5) with chosen optimizer (e.g., Gradient Descent).

### **Step 4: Visualize in TensorBoard**

Run in a new cell:

```bash
%load_ext tensorboard
%tensorboard --logdir logs/
```

This shows:

* Computation graph
* Loss curves
* Training speed comparison (CPU, GPU, TPU)

---

## **6. Comparing Device Performance**

The notebook contains benchmark cells to compare execution time using:

* **CPU**
* **GPU (Tesla T4)**
* **TPU (v5e-1)**

To test different runtimes:

1. In Colab, go to **Runtime → Change runtime type → Hardware accelerator**.
2. Select **CPU**, **GPU**, or **TPU**.
3. Rerun the training section and note execution time and MSE.

---

## **7. Customization**

You can modify these hyperparameters at the start of the notebook:

| Parameter       | Description                      | Default          |
| --------------- | -------------------------------- | ---------------- |
| `BATCH_SIZE`    | Batch size per step              | 1024             |
| `EPOCHS`        | Number of epochs                 | 5                |
| `LEARNING_RATE` | Learning rate for optimizer      | 0.001            |
| `HIDDEN_LAYERS` | List of neurons per hidden layer | [32, 16]         |
| `OPTIMIZER`     | Optimizer type                   | Gradient Descent |

Example:

```python
hidden_layers = [64, 32, 16]
learning_rate = 0.0005
```

---

## **8. Expected Output**

At the end of the run, you should observe:

* **Training loss decreasing** steadily per epoch.
* **Validation loss stabilizing**, showing no overfitting.
* **TensorBoard graph** visualizing full computation path.
* **Plots** comparing:

  * Epoch time vs number of hidden layers
  * Execution time for CPU vs GPU vs TPU

---

## **9. Notes and Tips**

* Use **small subsets** of the dataset (e.g., first 1M rows) for faster testing.
* TPU training requires the TFRecord input pipeline to be correctly configured.
* Ensure TensorBoard logs are written to different directories for CPU/GPU/TPU runs (e.g., `logs/cpu`, `logs/gpu`, etc.).
* For reproducibility, set the random seed (`RANDOM_SEED = 42`).

---

## **10. Author and Acknowledgments**

**Author:** Amar
**Institution:** IIT Madras
**Department:** Mathematics (M.Tech — Industrial Mathematics & Scientific Computing)

**Acknowledgments:**

* TensorFlow documentation and tutorials by Google.
* NYC Taxi & Limousine Commission (TLC) dataset available via Kaggle.

