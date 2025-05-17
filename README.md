# Effective Training of Neural Networks for Automatic Speech Recognition

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-orange)](https://huggingface.co/transformers)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)

This repository contains the source code, experiments, and resources for the Master's thesis: **"Effective Training of Neural Networks for Automatic Speech Recognition"**.

---

## Master Thesis Details
-   **Degree:** Master of Information Technology (NMAL)
-   **Institution:** [Faculty of Information Technology, Brno University of Technology (FIT BUT)](https://www.fit.vut.cz/.en)
-   **Author:** Matej Horník ([xhorni20@stud.fit.vut.cz](mailto:xhorni20@stud.fit.vut.cz))
-   **Supervisor:** Ing. Alexander Polok ([ipoloka@fit.vut.cz](mailto:ipoloka@fit.vut.cz))
-   **Year:** 2025
-   **Thesis Link:** [Official Thesis Page (VUT)](https://www.vut.cz/en/students/final-thesis/detail/164401) (Link will be fully active after defense)

---

## About This Project

This project systematically investigates efficient training strategies for encoder-decoder Transformer models in Automatic Speech Recognition (ASR). It explores initialization techniques, the role of adapter layers, Parameter-Efficient Fine-tuning (PEFT) methods like LoRA and DoRA, and the impact of domain-specific pre-training, primarily using the LibriSpeech and VoxPopuli datasets.

The code includes scripts for model creation, fine-tuning ( leveraging Hugging Face Transformers), evaluation, and implementations of various experimental setups discussed in the thesis.

---

## Key Outcomes & Model

A key result of this work is a Wav2Vec2-BART (base) model fine-tuned on English VoxPopuli, achieving a **Word Error Rate (WER) of 8.85%** on the test set.

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model_Card-matejhornik/wav2vec2--base_bart--base_voxpopuli--en-blue)](https://huggingface.co/matejhornik/wav2vec2-base_bart-base_voxpopuli-en)

You can find the model, along with usage instructions and a detailed model card, on the Hugging Face Hub:
[matejhornik/wav2vec2-base_bart-base_voxpopuli-en](https://huggingface.co/matejhornik/wav2vec2-base_bart-base_voxpopuli-en)

---

## Setup and Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management and packaging.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/hornikmatej/thesis_mit.git # Or your actual repo URL
    cd thesis_mit
    ```

2.  **Install Poetry:**
    If you don't have Poetry installed, follow the instructions on the [official Poetry website](https://python-poetry.org/docs/#installation).

3.  **Install dependencies:**
    This command will create a virtual environment and install all the necessary packages defined in `pyproject.toml` and `poetry.lock`.
    ```bash
    poetry install
    ```

4.  **Activate the virtual environment:**
    ```bash
    poetry shell
    ```
    You are now in the project's virtual environment with all dependencies available.

---

## Running Experiments

The repository is structured to facilitate the reproduction of experiments:
*   The main training script for sequence-to-sequence ASR models is `run_speech_recognition_seq2seq.py`.
*   Specific experiment configurations and launch commands are organized within shell scripts, primarily in the `run_scripts/` directory (e.g., `run_scripts/voxpopuli_best.sh`).
*   The `src/` directory contains custom modules for model creation, specialized trainers, data handling, etc.
*   Ensure you have the necessary datasets downloaded or accessible (e.g., via Hugging Face Datasets caching). Preprocessing scripts or arguments might be needed as detailed in the thesis or individual run scripts.

Please refer to the thesis document and the comments within the scripts for detailed instructions on running specific experiments.

---

## Citation

If you use code or findings from this thesis in your research, please consider citing:

[![CITE](https://excel.fit.vutbr.cz/wp-content/images/2023/FIT_color_CMYK_EN.svg)](https://www.vut.cz/en/students/final-thesis/detail/164401)

```bibtex
@mastersthesis{Hornik2025EffectiveTraining,
  author       = {Horník, Matej},
  title        = {Effective Training of Neural Networks for Automatic Speech Recognition},
  school       = {Brno University of Technology, Faculty of Information Technology},
  year         = {2025},
  supervisor   = {Polok, Alexander},
  type         = {Master's Thesis},
  note         = {Online. Available at: \url{https://www.vut.cz/en/students/final-thesis/detail/164401} and code at \url{https://github.com/hornikmatej/thesis_mit}}
}