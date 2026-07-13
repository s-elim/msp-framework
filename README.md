# Manipulation-Sufficient Perception (MSP) V5

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**MSP V5** is a state-of-the-art framework for robotic object pose estimation and grasping. Instead of relying on rigid, metric 3D reconstructions, MSP operates via **Manipulation-Sufficient Perception**. It leverages an Information Bottleneck approach to extract only the sufficient statistics required to guarantee physical success, discarding task-irrelevant metric details.

This repository is designed to be highly modular, scalable, and research-friendly, comparable to frameworks like Detectron2 and OpenVLA.

---

## 🌟 Key Features

*   **Variational Information Bottleneck (VIB):** Dynamically trades off state-compression rate with physical distortion outcomes (success, slip, margin) using custom loss formulations.
*   **Active Perception (Next-Best-View):** Amortized Value of Information (VoI) estimation to actively move the camera when epistemic uncertainty is high.
*   **Conformal Calibration:** Guarantees strict, distribution-free statistical bounds on execution success via Split Conformal Prediction and Adaptive Conformal Inference (ACI).
*   **Test-Time Adaptation (TTA):** Real-time belief updating via trust-region gradient descent following physical probe outcomes.
*   **Modular Architecture:** Swap out Backbones, Outcome Heads, and Physics Simulators effortlessly using the `@REGISTRY` pattern.

---

## 📂 Repository Structure

The codebase is strictly separated by logical concerns:

```text
msp_framework/
├── msp/
│   ├── core/         # VIB loss, Conformal calibration, TTA, Active Perception
│   ├── models/       # Encoders, Outcome Heads, and Vision Backbones
│   ├── data/         # Offline datasets, DataLoaders, and Augmentations
│   ├── physics/      # Physics Oracle (PyBullet/MuJoCo) & Action Samplers
│   ├── engine/       # PyTorch Trainer and Evaluator orchestrators
│   └── utils/        # Registries, deterministic seeders, and unified Loggers
├── scripts/          # Execution entry points (train.py, evaluate.py, deploy.py)
├── configs/          # Hydra YAML configurations (planned)
├── cookbook.md       # Detailed usage and training instructions
└── README.md         # This file
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/s-elim/msp-framework.git
cd msp-framework

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install core dependencies
pip install torch torchvision torchaudio
pip install hydra-core wandb numpy h5py scipy
```

### Usage

The framework is driven by execution scripts in the `scripts/` directory.

*   **Train the model:** `python scripts/train.py`
*   **Evaluate coverage bounds:** `python scripts/evaluate.py`
*   **Run interactive deployment:** `python scripts/deploy.py`

---

## 📚 Documentation

For a comprehensive guide on running training loops, overriding Hydra configurations, interpreting Weights & Biases logs, and extending the framework with custom backbones, please refer to the [**MSP V5 Cookbook**](./cookbook.md).

For theoretical underpinnings and complete mathematical formalization, see the original architecture documentation in the `Object Pose and Grasping` directory.

---

## 🤝 Contributing

This framework relies on a strict registry pattern. If you want to add a new vision backbone (e.g., Vision Transformers), simply inherit from `BaseBackbone`, decorate your class with `@BACKBONE_REGISTRY.register()`, and add the corresponding YAML configuration. 

## 📄 License
This project is licensed under the MIT License.