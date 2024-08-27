import tkinter as tk
import customtkinter as ctk
from torch.cuda.amp import autocast  # Updated import
from tkinter import filedialog
from PIL import Image, ImageTk



app = tk.Tk()
app.geometry("1200x720")
app.title("Stable Diffusion")
ctk.set_appearance_mode("blue")

logo = Image.open("logo.png")  # Replace 'logo.png' with the path to your logo image
logo = logo.resize((200, 200))  # Resize the logo to fit the desired area
logo_img = ImageTk.PhotoImage(logo)

logo_label = ctk.CTkLabel(master=app, image=logo_img, text="")
logo_label.place(x=35, y=-50)  # Adjust the position as needed

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



app.mainloop()