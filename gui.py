#!/usr/bin/env

from tkinter import *
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image
import PIL.Image as im
import os
import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler

MainWindow = Tk()
MainWindow.wm_title("Robotski vid - projekt čebelice")
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
        
'''
def open_img():
    filename = select_file()
    img = Image.open(filename)
    img = img.resize((1680, 900), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(MainWindow, image=img)
    panel.image = img
    panel.pack()
'''
def open_img():
    filename = select_file()
    iImage = np.array(im.open(filename))

    fig =plt.figure(figsize=(5,4), dpi=100)
    plt.imshow(iImage)

    canvas = FigureCanvasTkAgg(fig, master=mainarea)
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill= BOTH, expand=1)

    toolbar = NavigationToolbar2Tk(canvas, mainarea)
    toolbar.update()
    canvas.get_tk_widget(sidemenu=TOP, fill=BOTH, expand=1)


def _quit():
    MainWindow.quit()
    MainWindow.destroy()

btn = Button(sidemenu, text='Izberi sliko', command=open_img, width = 20, height = 3).place(x=120,y=50)
    
button = Button(master=mainarea, text="Quit", command=_quit)
button.pack(side=BOTTOM)

MainWindow.mainloop()
