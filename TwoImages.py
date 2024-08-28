import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
import torch
from diffusers import StableDiffusionPipeline
from tkinter import filedialog

# Initialize Stable Diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)


class StableDiffusionApp:
    def __init__(self):
        self.app = ctk.CTk()
        self.app.title("Stable Diffusion App")
        self.app.geometry("800x600")

        self.generated_images = []

        # Load and resize logo
        self.logo = Image.open("logo.png")
        self.logo = self.logo.resize((200, 200))
        self.logo_img = ImageTk.PhotoImage(self.logo)

        # Create UI elements
        self.logo_label = ctk.CTkLabel(self.app, image=self.logo_img, text="")
        self.prompt_label = ctk.CTkLabel(self.app, text="Enter Prompt:")
        self.prompt_textbox = ctk.CTkTextbox(self.app, height=100, width=580, corner_radius=10,
                                             font=("Times New Roman", 15), text_color="black", fg_color="white")
        self.generate_button = ctk.CTkButton(self.app, text="Generate", command=self.generate_images)
        self.save_button = ctk.CTkButton(self.app, text="Save", command=self.save_images)

        self.image_label1 = ctk.CTkLabel(self.app, height=256, width=256, text="", fg_color="gray", corner_radius=5)
        self.image_label2 = ctk.CTkLabel(self.app, height=256, width=256, text="", fg_color="gray", corner_radius=5)

        # Place UI elements
        self.logo_label.place(x=10, y=-50)
        self.prompt_label.pack(pady=(20, 0))
        self.prompt_textbox.pack(pady=10)
        self.generate_button.pack(pady=10)
        self.save_button.pack(pady=10)

        self.image_label1.pack(side="left", padx=10, pady=10)
        self.image_label2.pack(side="right", padx=10, pady=10)

    def generate_images(self):
        prompt_text = self.prompt_textbox.get("1.0", "end-1c")
        with torch.no_grad():
            images = pipe(prompt_text, num_images_per_prompt=2, guidance_scale=9.5).images

        self.generated_images = images
        self.display_images()

    def display_images(self):
        if len(self.generated_images) >= 2:
            img1 = ImageTk.PhotoImage(self.generated_images[0])
            self.image_label1.configure(image=img1)
            self.image_label1.image = img1

            img2 = ImageTk.PhotoImage(self.generated_images[1])
            self.image_label2.configure(image=img2)
            self.image_label2.image = img2

    def save_images(self):
        if not self.generated_images:
            print("No images to save")
            return

        # Save the first image
        file_path_1 = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if file_path_1:
            self.generated_images[0].save(file_path_1)

        # Save the second image
        file_path_2 = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if file_path_2:
            self.generated_images[1].save(file_path_2)

    def run(self):
        self.app.mainloop()


if __name__ == "__main__":
    app = StableDiffusionApp()
    app.run()