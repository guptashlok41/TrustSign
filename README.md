# TrustSign
### An Explainable and Adversarially Robust Traffic Sign Recognition System

TrustSign is a Trustworthy AI project that focuses on traffic sign recognition, model explainability, adversarial robustness, and prediction uncertainty.

The system uses three Convolutional Neural Network (CNN) models trained on the German Traffic Sign Recognition Benchmark (GTSRB) dataset and provides tools to evaluate their performance and trustworthiness.

## Key Features

- **Traffic Sign Classification:** Recognizes 43 different traffic sign classes.
- **Multiple CNN Models:** Compares Original CNN, PGD Adversarial CNN, and Mixed Adversarial CNN.
- **Confidence Analysis:** Displays prediction confidence and normalized entropy.
- **Explainable AI:** Uses Grad-CAM to visualize model attention.
- **Adversarial Robustness:** Evaluates models using FGSM and PGD attacks.
- **Trustworthiness Analysis:** Examines prediction uncertainty and high-confidence incorrect predictions.
- **Calibration Analysis:** Evaluates model calibration using Expected Calibration Error (ECE).
- **GPU Acceleration:** Supports CUDA-enabled NVIDIA GPUs.

## Models

1. Original CNN
2. PGD Adversarial CNN
3. Mixed Adversarial CNN

Each model is evaluated on clean traffic sign images and adversarial examples.

## Dataset

**German Traffic Sign Recognition Benchmark (GTSRB)**

- 43 traffic sign classes
- 39,209 training images
- 12,630 test images

The dataset is not included in this repository. Place the downloaded dataset in the following directory:

```text
data/gtsrb/
```

## Technology Stack

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- OpenCV
- Pillow

## Project Structure

```text
TrustSign/
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── model.py
│   ├── inference.py
│   ├── gradcam.py
│   ├── adversarial_attacks.py
│   └── trustworthiness.py
│
├── models/
│   ├── original_cnn.pth
│   ├── pgd_adversarial_cnn.pth
│   └── mixed_adversarial_cnn.pth
│
├── results/
│   ├── graphs/
│   └── experimental_results
│
├── custom_images/
│
└── data/
    └── gtsrb/
```

## Installation

Clone the repository:

```bash
git clone https://github.com/guptashlok41/TrustSign.git
cd TrustSign
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Make sure the GTSRB dataset is available in `data/gtsrb/`.

## Running the Application

Run the main application:

```bash
python main.py
```

The application provides the following options:

1. Predict Traffic Sign
2. Grad-CAM Explainability
3. Adversarial Robustness Results
4. Trustworthiness Analysis
5. Calibration Analysis
6. Overall Experimental Results
7. Exit

## Experimental Results

| Model | Clean Test Accuracy |
|---|---:|
| Original CNN | 93.89% |
| PGD Adversarial CNN | 80.89% |
| Mixed Adversarial CNN | 90.13% |

### Adversarial Evaluation

The models are evaluated against:

- Fast Gradient Sign Method (FGSM)
- Projected Gradient Descent (PGD)

Multiple perturbation strengths are used to study the effect of adversarial attacks on model performance.

### Explainability

Grad-CAM is used to generate visual heatmaps that provide approximate explanations of model predictions.

### Calibration

Temperature Scaling is evaluated using Expected Calibration Error (ECE).

The experimental results showed:

- ECE before calibration: 0.0236
- ECE after calibration: 0.0238

Temperature scaling did not improve calibration in this experiment.

## Limitations

- Performance on real-world traffic sign images may differ from performance on GTSRB test images.
- High prediction confidence does not guarantee correctness.
- Grad-CAM provides approximate visual explanations rather than definitive evidence of model reasoning.
- Trustworthiness categories use project-defined heuristic thresholds.

## Author

**Shlok Gupta**

B.Tech – Computer Science and Engineering

ABV-IIITM Gwalior

## License

This project is developed for academic and educational purposes.