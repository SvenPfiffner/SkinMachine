# SkinMachine

SkinMachine is a simple diffusion model to generate novel Minecraft skins at random. It consists of a simple UNet diffusion model trained on a dataset of roughly 900'000 minecraft skins.

## Example outputs


### Prerequisites

- Build on python 3.10
- Required Python libraries (see `requirements.txt`)

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/SvenPfiffner/SkinMachine
    cd SkinMachine
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

- ```train.ipynb``` contains the training code of the model. Fine tune, extend or adjust to your liking to make your own model
- ```skindataset.py``` a simple dataset class to load the skin dataset
- ```skindiffuser.py``` implementation of the UNet and a simple NoiseScheduler for the diffusion forward pass
- ```infer.ipynb``` contains inference code to generate new skins using the trained model

### Additional data

**Training**: To train your own version of the model, download the dataset and pass it to the dataloader. Download link for the dataset is given in the *Acknowledgements* section.

**Inference**: If you just want to try inference without training on your own, you can directly use the model checkpoint found under ```checkpoints/``` by running the infer.ipynb script

## Contributing

Contributions are welcome! Feel free to use this project as a starting point for your personal creations!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The code, especially the architecture implementation, is largely based on and inspired by the amazing Youtube tutorial by **DeepFindr**. See: https://www.youtube.com/watch?v=a4Yfz2FxXiY

- The dataset used for this project was sourced from kaggle and is provided at https://www.kaggle.com/datasets/sha2048/minecraft-skin-dataset/data by SHA2048.