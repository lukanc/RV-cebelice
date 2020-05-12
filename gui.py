#!/usr/bin/env

from tkinter import *
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image
import os

MainWindow = Tk()
MainWindow.geometry("1280x900")
MainWindow.resizable(width=True, height=True)
# stranski menu z parametri ?
sidemenu = Frame(MainWindow, width=400, bg='white', height=500, relief='sunken', borderwidth=2)
sidemenu.pack(expand=False, fill='both', side='left', anchor='nw')

# image dispaly label
mainarea = Frame(MainWindow, bg="#CCC", width=500, height=500)
mainarea.pack(expand=True, fill='both', side='right')
scrollbar = Scrollbar(mainarea)
scrollbar.pack(side=RIGHT, fill=Y)

def select_file():
    while True:
        filename = filedialog.askopenfilename(title='open')
        extension = os.path.splitext(filename)[1]
        if extension in ('.jpg', '.jpeg', ".png", ".tiff", ".gif"):
            return filename
        else:    
            answer = messagebox.askquestion( "Napačna datoteka", "Izbrana datoteka ni slika! \n\nŽeliš izbrati novo datoteko?")
            if answer == "no":
                break
        

def open_img():
    filename = select_file()
    img = Image.open(filename)
    img = img.resize((1680, 900), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(MainWindow, image=img)
    panel.image = img
    panel.pack()

btn = Button(sidemenu, text='Izberi sliko', command=open_img, width = 20, height = 3).place(x=120,y=50)


MainWindow.mainloop()
