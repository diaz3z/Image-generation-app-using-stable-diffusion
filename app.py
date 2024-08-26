import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
import torch
from torch.cuda.amp import autocast  # Updated import
from diffusers import StableDiffusionPipeline

app = tk.Tk()
app.geometry("532x632")
app.title("Mohammad Zaid")
ctk.set_appearance_mode("dark")

promt = ctk.CTkEntry(master=app, height=40, width=512, corner_radius=10, font=("Arial", 20), text_color="black", fg_color="white")
promt.place(x=10, y=10)

lmain = ctk.CTkLabel(master=app, height=512, width=512, text="")  # Change to CTkLabel for displaying images
lmain.place(x=10, y=110)

trigger = ctk.CTkButton(master=app, height=40, width=120, corner_radius=10, font=("Arial", 20), text_color="white", fg_color="gray", text="Generate", command=lambda: generate())  # Add command to button
trigger.place(x=206, y=60)

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

def generate():
    prompt_text = promt.get()
    with autocast():  # No need to specify the device here
        image = pipe(prompt_text, guidance_scale=8.5).images[0]  # Access the image correctly

    img = ImageTk.PhotoImage(image)  # Convert PIL Image to Tkinter-compatible format
    lmain.configure(image=img)  # Display the image
    lmain.image = img  # Keep a reference to the image

app.mainloop()
