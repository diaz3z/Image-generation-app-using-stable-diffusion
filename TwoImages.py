import tkinter as tk
import customtkinter as ctk
from torch.cuda.amp import autocast  # Updated import
from PIL import Image, ImageTk
import torch
from diffusers import StableDiffusionPipeline
from tkinter import filedialog

app = tk.Tk()
app.geometry("1200x720")
app.title("Stable Diffusion")
ctk.set_appearance_mode("blue")

logo = Image.open("logo.png")
logo = logo.resize((200, 200))
logo_img = ImageTk.PhotoImage(logo)

logo_label = ctk.CTkLabel(master=app, image=logo_img, text="")
logo_label.place(x=35, y=-50)

border_frame = ctk.CTkFrame(master=app, height=124, width=516, fg_color="cadetblue3", corner_radius=12)
border_frame.place(x=358, y=8)

promt = ctk.CTkTextbox(master=border_frame, height=120, width=512, corner_radius=10, font=("Times New Roman", 15), text_color="black", fg_color="white")
promt.place(x=2, y=2)

lmain1 = ctk.CTkLabel(master=app, height=512, width=512, text="", fg_color="gray",corner_radius=5)
lmain1.place(x=58,y=150)

lmain2 = ctk.CTkLabel(master=app, height=512, width=512, text="", fg_color="gray",corner_radius=5)
lmain2.place(x=638,y=150)

trigger = ctk.CTkButton(master=app, height=40, width=100, corner_radius=10, font=("Times New Roman Baltic", 20), text_color="black", fg_color="cadetblue3", text="Generate",  command=lambda: generate())
trigger.place(x=920, y=55)

save_button = ctk.CTkButton(master=app, height=30, width=120, corner_radius=7, font=("Times New Roman Baltic", 20), text_color="black", fg_color="skyblue", text="Save", command=lambda: save_images())
save_button.place(x=540, y=670)

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

generated_images = []


def generate():
    promt_text = promt.get("1.0", "end-1c")
    with autocast():
        image = pipe(promt_text, num_images_per_prompt=2,guidance_scale=9.5).images

    global generated_images
    generated_images = image

    img1 = ImageTk.PhotoImage(image[0])
    lmain1.configure(image=img1)
    lmain1.image = img1

    img2 = ImageTk.PhotoImage(image[1])
    lmain2.configure(image=img2)
    lmain2.image = img2


def save_images():
    if not generated_images:
        print("No images to save")
        return

    # Save the first image
    file_path_1 = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
    if file_path_1:
        generated_images[0].save(file_path_1)

    # Save the second image
    file_path_2 = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
    if file_path_2:
        generated_images[1].save(file_path_2)

app.mainloop()

