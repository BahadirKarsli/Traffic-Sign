<div align="center">
  <h1>🚦 Traffic Sign Recognition System</h1>
  <p>A Real-Time Classification System based on PyTorch and CNN for Autonomous Vehicles and Advanced Driver Assistance Systems (ADAS)</p>
</div>

<br/>

## 🎯 About the Project

This project is a **Deep Learning** study designed to minimize driver errors and enhance the environmental perception of autonomous driving systems. Built with **PyTorch**, the **Convolutional Neural Network (CNN)** model detects and classifies roadside traffic signs with high accuracy.

## 📊 Dataset

The model was trained and tested using the **[German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/)** dataset. This dataset consists of tens of thousands of traffic sign images captured under various lighting, angle, and weather conditions.

## 🏆 Performance Metrics

The developed CNN model achieved highly successful results on the test dataset:
- **Test Accuracy:** 95.95%
- **Precision:** 0.96
- **Recall:** 0.96
- **F1-Score:** 0.96

## 📂 Repository Structure

The current directory structure of this repository is as follows:

```text
├── notebooks/       # Jupyter Notebooks containing Exploratory Data Analysis (EDA), model training, and evaluation
├── python_files/    # Python source codes including model architecture, data preprocessing, and helper scripts
├── report/          # Documentation and reports detailing the project development process and findings
├── test_images/     # Sample images used to independently test the model's inference capabilities
└── README.md        # Project documentation
```

## 🛠️ Technologies Used
Programming Language: Python

Deep Learning Framework: PyTorch (CNN Architecture)

Image Processing: OpenCV, PIL (Python Imaging Library)

Data Analysis & Visualization: NumPy, Pandas, Matplotlib, Seaborn

## 🚀 Installation
Follow the steps below to run the project on your local machine:

Clone the Repository:

```
git clone [https://github.com/BahadirKarsli/Traffic-Sign.git](https://github.com/BahadirKarsli/Traffic-Sign.git)
cd Traffic-Sign
```
Install Required Dependencies:
It is recommended to use a virtual environment. Install the necessary packages via:

```
pip install torch torchvision numpy pandas matplotlib opencv-python jupyter
```
## 💻 Usage
1. Model Training and Analysis:
Navigate to the notebooks/ directory and open the .ipynb files via Jupyter Notebook. You can run the cells step-by-step to perform dataset exploration, data augmentation, and model training.

2. Inference (Prediction):
You can test the model's performance on the sample images located in the test_images/ directory or provide your own traffic sign images using the relevant scripts located in the python_files/ directory.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page or submit a Pull Request if you want to contribute to the project.
