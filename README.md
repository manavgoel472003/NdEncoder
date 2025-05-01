# NdEncoder

An image auto-encoder built using a spatially aware linear layer (`NdLinear`) to preserve and exploit multi-dimensional structure in feature maps. 

---

## Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Project Structure](#project-structure)  
- [Model Architecture](#model-architecture)  
- [Examples & Notebook](#examples--notebook)  
- [Dependencies](#dependencies)  
- [Contributing](#contributing)  
- [License](#license)  
- [Contact](#contact)  

---

## Overview

NdEncoder demonstrates how to build an image auto-encoder that leverages the **NdLinear** layer from the `ndlinear` package. Instead of flattening spatial inputs for standard fully-connected layers, NdLinear aligns and projects along each spatial axis—yielding richer, parameter-efficient feature representations.

---

## Features

- **Spatially Aware Layers**  
  Use `NdLinear` to maintain spatial structure in both encoder and decoder.  
- **Convolutional Upsampling**  
  Final `ConvTranspose2d` layer to reconstruct high-resolution images.  
- **Flexible API**  
  Modular `Encoder` / `Decoder` classes defined in `models.py`.  
- **Training & Evaluation**  
  Command-line script (`train_eval.py`) for training, validation, and checkpointing.  
- **Interactive Demo**  
  Jupyter notebook (`auto_encoder.ipynb`) showcasing data loading, training loop, and reconstructions.

---

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/manavgoel472003/NdEncoder.git
   cd NdEncoder
   ```

2. **Create & activate your virtual environment**  
   ```bash
   python -m venv venv
   source venv/bin/activate      # on Linux/macOS
   venv\Scripts\activate.bat   # on Windows
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Run the Notebook

Launch Jupyter and open the demo:
```bash
jupyter notebook auto_encoder.ipynb
```
- Walks through data loading, model instantiation, training loop, and image reconstructions.
---

## Project Structure

```
├── README.md
├── requirements.txt
├── auto_encoder.ipynb    # Interactive demo notebook
├── models.py             # Definition of NdEncoder / NdDecoder classes
├── train_eval.py         # CLI script for training & evaluation
└── checkpoints/          # (auto-created) saved model weights & logs
```

---

## Model Architecture

- **Encoder**  
  Stacks of `NdLinear` + `ReLU` + `BatchNorm` layers progressively reduce spatial dimensions and increase feature depth.

- **Latent Space**  
  A compact multi-dimensional tensor (e.g. `B × C_latent × H’ × W’`) capturing the image’s key features.

- **Decoder**  
  Mirrors the encoder using `NdLinear` blocks, followed by a final `ConvTranspose2d` to upscale back to original resolution.

See `models.py` for full layer definitions and hyperparameters.

---

## Examples & Notebook

- **auto_encoder.ipynb** demonstrates:
  - Loading an image dataset via `torchvision.datasets`  
  - Custom `Subset`/`DataLoader` setup  
  - Training loop with `torch.optim.Adam` & `MSELoss`  
  - Visualization of original vs. reconstructed images  

---

## Dependencies

- `torch>=2.0.0`  
- `torchvision>=0.15.0`  
- `ndlinear`  
- `tqdm>=4.65.0`  
- `matplotlib>=3.7.0`  

---

## Contributing

Contributions and issue reports are very welcome!  
1. Fork the repo  
2. Create a feature branch  
3. Commit your changes & push  
4. Open a Pull Request detailing your improvements  

---

## License

This project does not currently include a license file.  
If you’d like to use or extend it, please contact the maintainer to clarify usage terms.

---

## Contact

**Manav Goel**  
- GitHub: [@manavgoel472003](https://github.com/manavgoel472003)  
- Email: _manavgoel47@gmail.com_

Feel free to open issues or reach out for questions!

