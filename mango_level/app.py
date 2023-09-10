#pip install Pillow
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os 
from aimodel import AI_model

class ImageViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("芒果等級辨識")
        self.root.geometry("500x400")
        self.create_gui()

    def create_gui(self):
        label_font = ("Helvetica", 24)  
        self.browse_button = tk.Button(self.root, text="上傳芒果圖片", command=self.browse_image,font=label_font)
        self.canvas = tk.Canvas(self.root, width=240, height=210) 
        self.file_name_label = tk.Label(self.root, text="", font=label_font)
        self.browse_button.grid(row=0, column=0, padx=10, pady=10)
        self.canvas.grid(row=1, column=0, padx=10, pady=10)
        self.file_name_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")

    def browse_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.gif")])
        if file_path:
            self.display_image(file_path)


    def display_image(self, image_path):
        image = Image.open(image_path)
   
        fixed_size = (225, 225)
        image = image.resize(fixed_size) 
        
        photo = ImageTk.PhotoImage(image)

        
        self.canvas.delete("all")  
        self.canvas.create_image(0, 0, anchor="nw", image=photo)
        self.canvas.image = photo  
        
        
        conf,label=model.predict(image_path)
        print('Class:', label, end='')
        print('Confidence score:', conf)
        conf=round(conf,3)
        
        self.file_name_label.config(text="等級"f"{label}({conf})")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageViewerApp(root)
    
    model_dir='model'
    model_file='ResNet101V2.h5'
    label_file='labels.txt'
          
    model_file=os.path.join(model_dir,model_file)
    class_file=os.path.join(model_dir,label_file)
    model=AI_model(model_file,class_file)
    
   
    
    root.mainloop()
    

