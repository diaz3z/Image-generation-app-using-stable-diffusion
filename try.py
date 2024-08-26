import tkinter as tk
import customtkinter as ctk
from PIL import ImageTk
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline


app = tk.Tk()
app.geometry("532x632")
app.title("Mohammad Zaid")
ctk.set_appearance_mode("dark") 

promt = ctk.CTkEntry(master=None, height=40, width=512,corner_radius=10,font=("Arial", 20), text_color="black", fg_color="white")
promt.place(x=10, y=10)

lmain = ctk.CTkEntry(master=None,height=512, width=512)
lmain.place(x=10, y= 110)

trigger = ctk.CTkButton(master=None, height=40, width=120,corner_radius=10,font=("Arial", 20), text_color="white", fg_color="gray" )
trigger.configure(text="Generate")
trigger.place(x= 206, y=60)

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

def generate():
    with autocast(device):
        image = pipe(promt.get(), guidance_scale=8.5)["sample"][0]

    img = ImageTk.PhotoImage(image)
    img.save('generated.png')
    lmain.configure(image=img)


app.mainloop()