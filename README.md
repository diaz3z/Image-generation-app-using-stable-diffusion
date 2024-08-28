# Stable Diffusion Image Generator

This project is a graphical user interface (GUI) application that utilizes the Stable Diffusion model to generate images from text prompts. It allows users to input text descriptions and generate two corresponding images using the power of AI.

## Features

- User-friendly GUI built with Tkinter and CustomTkinter
- Text-to-image generation using Stable Diffusion v1.4
- Generates two images simultaneously for each prompt
- Option to save generated images

## Requirements

- Python 
- PyTorch
- Tkinter
- CustomTkinter
- Pillow
- Diffusers

## Installation

1. Clone this repository:

```bash
git clone https://github.com/diaz3z/Image-generation-app-using-stable-diffusion.git

```
2. Install the required packages:
```bash
pip install -r requirements.txt

```
3. Make sure you have CUDA installed for GPU acceleration (recommended for faster image generation).

## Usage

1. Run the script:
```bash
python TwoImages.py

```
2. Enter your text prompt in the text box.

3. Click the "Generate" button to create two images based on your prompt.

4. Use the "Save" button to save the generated images to your local machine.
![Mohammad Zaid](https://github.com/user-attachments/assets/26975fa3-e2fb-4ca7-8f47-9dea624e76be)

## How it works

1. The application creates a GUI window using Tkinter and CustomTkinter.
2. It loads the Stable Diffusion v1.4 model from HuggingFace's model hub.
3. When you enter a prompt and click "Generate", the app uses the Stable Diffusion pipeline to create two images.
4. The generated images are displayed in the GUI.
5. You can save the images using the "Save" button, which opens a file dialog for each image.

## Screenshots
![2](https://github.com/user-attachments/assets/9a82a490-a7cb-458c-adcc-2dc584e3e8a1)
![3](https://github.com/user-attachments/assets/3df20ddf-73cf-4fea-9610-bce098ad21f5)
![4](https://github.com/user-attachments/assets/65dfbb1b-72ac-4c41-8afd-0fbe3501d7e7)
![5](https://github.com/user-attachments/assets/9a348f9c-4f28-408c-b52f-1760373ec4d0)
![6](https://github.com/user-attachments/assets/c4ebf7d6-be98-4121-9b71-5b3c9a6646d3)
![7](https://github.com/user-attachments/assets/db0f92b6-0a3e-4d85-acfb-16382c2fe6f3)


## Note

This application requires a CUDA-capable GPU for efficient image generation. Make sure you have the appropriate hardware and drivers installed.

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check [issues page](https://github.com/diaz3z/Image-generation-app-using-stable-diffusionr/issues) if you want to contribute.

## License

[MIT](https://choosealicense.com/licenses/mit/)
