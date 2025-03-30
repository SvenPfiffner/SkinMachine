# SkinMachine

SkinMachine is a simple diffusion model to generate novel Minecraft skins at random. It consists of a simple UNet diffusion model trained on a dataset of roughly 900'000 minecraft skins.

## Results

Despite the relatively simple architecture and brief training period (the denoising model was trained for only 20 epochs, achieving an L1 loss of approximately 0.05), the model is capable of producing visually interesting and reasonably usable skins. The use of a self-attention mechanism in the bottleneck of the U-Net contributes to the overall coherence in color and theme across each generated output. With extended training or the addition of text-based conditioning, the model could show potential for further improvement.

### Loss Graph

### Inference results
Below are three rendered examples of skins produced with the provided checkpoint of only 20 epochs

![example1](https://github.com/SvenPfiffner/SkinMachine/blob/main/output/render_1.png)
![example2](https://github.com/SvenPfiffner/SkinMachine/blob/main/output/render_2.png)
![example3](https://github.com/SvenPfiffner/SkinMachine/blob/main/output/render_3.png)

## About the code

### Prerequisites

- Build on python 3.10
  
### Usage

- ```train.ipynb``` contains the training code of the model. Fine tune, extend or adjust to your liking to make your own model
- ```skindataset.py``` a simple dataset class to load the skin dataset
- ```skindiffuser.py``` implementation of the UNet and a simple NoiseScheduler for the diffusion forward pass
- ```infer.ipynb``` contains inference code to generate new skins using the trained model

### Additional data

**Training**: To train your own version of the model, download the dataset and pass it to the dataloader. Download link for the dataset is given in the *Acknowledgements* section.

**Inference**: If you just want to try inference without training on your own, you can directly use the model checkpoint provided on [HuggingFace🤗](https://huggingface.co/SvenPfiffner/SkinMachine/tree/main)

## Contributing

Contributions are welcome! Feel free to use this project as a starting point for your personal creations!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The code, especially the architecture implementation, is largely based on and inspired by the amazing Youtube tutorial by **DeepFindr**. See: https://www.youtube.com/watch?v=a4Yfz2FxXiY

- The dataset used for this project was sourced from kaggle and is provided at https://www.kaggle.com/datasets/sha2048/minecraft-skin-dataset/data by SHA2048.
