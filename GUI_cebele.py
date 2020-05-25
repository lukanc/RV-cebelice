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


##Definicija funkcij

def select_file():
    while True:
        filename = filedialog.askopenfilename(title='open')
        extension = os.path.splitext(filename)[1]
        if extension in ('.jpg', '.jpeg', ".png", ".tiff", ".gif"):
            return filename
        else:
            answer = messagebox.askquestion("Napačna datoteka",
                                            "Izbrana datoteka ni slika! \n\nŽeliš izbrati novo datoteko?")
            if answer == "no":
                break


def open_img():
    filename = select_file()
    global iImage
    iImage = np.array(im.open(filename))
    global fig
    fig = plt.figure(figsize=(5, 4), dpi=100)
    plt.imshow(iImage)
    global canvas
    global toolbar
    canvas = FigureCanvasTkAgg(fig, master=mainarea)
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

    toolbar = NavigationToolbar2Tk(canvas, mainarea)
    toolbar.update()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

def oznake3x3():
    global coords
    coords = []

    global cid
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print(ix, iy)

    global coords
    coords.append((ix, iy))

    # kaj želimo pokazati na platnu
    plt.imshow(iImage)
    plt.plot(ix, iy, 'or', markersize=5.0)

    # posodobi platno/sliko
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

    global cid
    if len(coords) == 8:
        fig.canvas.mpl_disconnect(cid)
    return coords

def izpis_tock():
    global array_coords
    array_coords = np.array(coords)
    print(array_coords.shape)
    print(array_coords)

    #kaj želimo pokazati na platnu
    plt.imshow(iImage)
    plt.plot(array_coords[:, 0], array_coords[:, 1], 'or', markersize=5.0)

    #posodobi platno/sliko
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

def _quit():
    global MainWin
    okno.quit()
    okno.destroy()

###################################
class CebeleGUI:
    def __init__(self, MainWin):
        self.master = MainWin
        MainWin.title("Robotski vid - projekt čebelice")
        MainWin.geometry("1280x900")
        MainWin.resizable(width=True, height=True)
        # stranski menu z parametri ?
        sidemenu = Frame(MainWin, width=400, bg='white', height=500, relief='sunken', borderwidth=2)
        sidemenu.pack(expand=False, fill='both', side='left', anchor='nw')

        # image dispaly label
        global mainarea
        global scrollbar
        mainarea = Frame(MainWin, bg="#CCC", width=500, height=500)
        mainarea.pack(expand=True, fill='both', side='right')
        scrollbar = Scrollbar(mainarea)
        scrollbar.pack(side=RIGHT, fill=Y)

        #### BUTTONS ####
        ## 1
        self.ime_datoteke_button = Button(sidemenu, text='Izberi sliko', command=self.open_image, width=20, height=3)
        self.ime_datoteke_button.place(x=120, y=50)

        ## 2
        self.button_oznake3x3 = Button(sidemenu, text='Oznaci tocke', command=self.potrdi_oznake3x3, width=20, height=3)
        self.button_oznake3x3.place(x=120, y=120)

        ## 3
        self.button_izpis_tock = Button(sidemenu, text='Izpisi tocke', command=self.potrdi_izpis_tock, width=20, height=3)
        self.button_izpis_tock.place(x=120, y=190)

        ## 4
        self.button_quit = Button(master=mainarea, text="Quit", command=self.potrdi_quit)
        self.button_quit.pack(side=BOTTOM)

    ### Definiranje kaj izvede vsak gumb ###
    def open_image(self):
        open_img()
        self.izpis.insert(END, 'Sprejet korak:\n')

    def potrdi_oznake3x3(self):
        oznake3x3()

    def potrdi_izpis_tock(self):
        izpis_tock()

    def potrdi_quit(self):
        _quit()





okno = Tk()
my_gui = CebeleGUI(okno)
okno.mainloop()
