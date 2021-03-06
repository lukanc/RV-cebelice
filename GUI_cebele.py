#!/usr/bin/env

from tkinter import *
from tkinter import filedialog, messagebox
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from knjiznica import *
from PIL import Image, ImageOps



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
    global filename
    filename = select_file()
    global iImage
    iImage = np.array(im.open(filename))
    # print('tip ob odpiranju', iImage.dtype)

    ##pocistimo prejsno sliko ki je pikazana
    plt.clf()
    # kaj želimo pokazati na platnu
    plt.imshow(iImage)

    # posodobi platno/sliko
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)


def oznake3x3():
    global coords
    coords = []

    global cid
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    # print(ix, iy)

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
    # print(array_coords.shape)
    # print(array_coords)

    ##pocistimo prejsno sliko ki je pikazana
    plt.clf()

    #kaj želimo pokazati na platnu
    plt.imshow(iImage)
    plt.plot(array_coords[:, 0], array_coords[:, 1], 'or', markersize=5.0)

    #posodobi platno/sliko
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

def porovnava():
    global iImage
    # print('tip v porovnavi:', iImage.dtype)
    iImage = np.asanyarray(iImage)
    iCalImageG = colorToGray(iImage)
    # koordinate v prostoru slike
    # print(array_coords.shape)

    iCoorU = array_coords[:, 0].flatten()
    iCoorV = array_coords[:, 1].flatten()
    ##velikost satja

    y_smer_velikost_satja = 3840
    x_smer_velikost_satja =2260

    # koordinate kalibra v metricnem prostoru
    iCoorX = np.array([0, x_smer_velikost_satja/2, x_smer_velikost_satja, x_smer_velikost_satja, x_smer_velikost_satja, x_smer_velikost_satja/2, 0, 0])
    iCoorY = np.array([0, 0, 0, y_smer_velikost_satja/2, y_smer_velikost_satja, y_smer_velikost_satja, y_smer_velikost_satja, y_smer_velikost_satja/2])

    k = 1
    iCoorX= iCoorX * k
    iCoorY = iCoorY * k

    # oblikuj koordinate v matrike (Nx2)
    ptsUV = np.vstack((iCoorU, iCoorV)).transpose()
    ptsXY = np.vstack((iCoorX, iCoorY)).transpose()

    # doloci zacetni priblizek parametrov
    oMatA = mapAffineApprox2D(ptsUV, ptsXY)

    # preslikava z afinim priblizkom
    Uc = iCalImageG.shape[1] / 2
    Vc = iCalImageG.shape[0] / 2
    iParAffine = np.array([oMatA[0, 0], oMatA[0, 1], oMatA[0, 2],
                           oMatA[1, 0], oMatA[1, 1], oMatA[1, 2],
                           0, 0, Uc, Vc, 0])

    from scipy.optimize import fmin
    F = lambda x: geomCalibErr(x, iCoorU, iCoorV, iCoorX, iCoorY)
    iParOpt = fmin(func=F, x0=iParAffine, maxiter=8000, disp=1, xtol=1e-6,
                   ftol=1e-6, maxfun=None)

    iCoorXx, iCoorYy = np.meshgrid(range(iImage.shape[1]),
                                   range(iImage.shape[0]),
                                   sparse=False, indexing='xy')

    # print("XX:", iCoorXx.shape)
    # print("YY", iCoorYy.shape)
    global Calibimage
    Calibimage = geomCalibImageRGB(iParOpt, iImage, iCoorXx, iCoorYy, 1)
    # global canvas
    # canvas.delete()
    ##384 x 226 mm velikost panja

    # kaj želimo pokazati na platnu
    plt.clf()
    plt.imshow(Calibimage)

    # posodobi platno/sliko
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
    iImage = Calibimage
    # print(iImage.dtype)

def razgradnja_slike_ai():
    global iImage
    global pokritost_cebel
    global pokritost_cebel_procent
    iImage = np.asanyarray(iImage).astype(np.uint8)

    preds_test_t = razgradnja(iImage)
    pokritost_cebel = np.count_nonzero(preds_test_t)  ##poda st pixlov ki jih pokrivajo cebele
    st_pixlu = (iImage.size) / 3
    pokritost_cebel_procent = pokritost_cebel/st_pixlu
    # print('stevilo pixlov Unet', np.count_nonzero(preds_test_t))

def savefile():
    filename_save = filedialog.asksaveasfilename(defaultextension=".png")
    if not filename_save:
        return
    Calibimage.astype(np.uint8)
    Calibimage_shrani = np.asanyarray(Calibimage)

    im = Image.fromarray((Calibimage_shrani * 255).astype(np.uint8))
    im = ImageOps.invert(im)
    im.save(filename_save)

def rotation(i):
    global iImage
    iImage = Image.open(filename)
    iImage = iImage.rotate(i, expand=1)
    ##pocistimo prejsno sliko ki je pikazana
    plt.clf()
    # kaj želimo pokazati na platnu
    plt.imshow(iImage)

    # posodobi platno/sliko
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)


def izracun_st_cebel():
    global iImage
    global st_cebel

    ##stevilo vseh pixlov
    st_pixlu = (iImage.size)/3
    # print(st_pixlu)
    povrsina_cebele = 9500
    # print(povrsina_cebele)

    st_cebel = pokritost_cebel/povrsina_cebele

    print('stevilo cebel:', st_cebel)


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
        sidemenu = Frame(MainWin, width=400, bg='white', height=600, relief='sunken', borderwidth=2)
        sidemenu.pack(expand=False, fill='both', side='left', anchor='nw')


        # image dispaly label
        global mainarea
        global scrollbar
        mainarea = Frame(MainWin, bg="#CCC", width=800, height=500)
        mainarea.pack(expand=True, fill='both', side='right')
        scrollbar = Scrollbar(mainarea)
        scrollbar.pack(side=RIGHT, fill=Y)

        global fig
        fig = plt.figure(figsize=(5, 4), dpi=100)
        global canvas
        global toolbar
        canvas = FigureCanvasTkAgg(fig, master=mainarea)
        canvas.draw()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

        toolbar = NavigationToolbar2Tk(canvas, mainarea)
        toolbar.update()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

        x_smer_gumbi = 120
        razmak = 60

        #### BUTTONS ####
        ## 1
        self.ime_datoteke_button = Button(sidemenu, text='Izberi sliko', command=self.open_image, width=20, height=3)
        self.ime_datoteke_button.place(x=x_smer_gumbi, y=razmak*0.1)

        ## 2
        self.ime_st_rotacij = Label(sidemenu, text="Vnesi stopinje rotacije").place(x=x_smer_gumbi, y=razmak*1)

        ## 3
        self.vnos_st_rotacij = Entry(sidemenu)
        self.vnos_st_rotacij.place(x=x_smer_gumbi, y=razmak*1.5)

        ## 4
        self.button_rotate_img = Button(sidemenu, text="Zarotiraj sliko", command=self.rotate, width=20, height=3)
        self.button_rotate_img.place(x=x_smer_gumbi, y=razmak*2)

        ## 5
        self.button_oznake3x3 = Button(sidemenu, text='Oznaci tocke', command=self.potrdi_oznake3x3, width=20, height=3)
        self.button_oznake3x3.place(x=x_smer_gumbi, y=razmak*3)

        ## 7
        self.button_porovnava = Button(sidemenu, text='Izvedi poravnavo', command=self.potrdi_porovnava, width=20, height=3)
        self.button_porovnava.place(x=x_smer_gumbi, y=razmak*4)

        ## 8
        self.button_save_img = Button(sidemenu, text="Shrani sliko", command=self.save_img, width=20, height=3)
        self.button_save_img.place(x=x_smer_gumbi, y=razmak * 5)

        ## 9
        self.button_maska = Button(sidemenu, text="Izracun st cebel", command=self.potrdi_maska, width=20, height=3)
        self.button_maska.place(x=x_smer_gumbi, y=razmak*6)

        ## END
        self.izpis = Text(sidemenu, width=50, height=30)
        self.izpis.place(x=x_smer_gumbi/5, y=razmak * 8)

        ## QUIT
        self.button_quit = Button(master=mainarea, text="Quit", command=self.potrdi_quit)
        self.button_quit.pack(side=BOTTOM)



    ### Definiranje kaj izvede vsak gumb ###
    def open_image(self):
        open_img()
        self.izpis.insert(END, 'Odprta slika\n')
        self.izpis.insert(END, filename)
        self.izpis.insert(END, '\n')

    def save_img(self):
        savefile()
        self.izpis.insert(END, 'Slika je shranjena\n')

    def potrdi_oznake3x3(self):
        oznake3x3()
        self.izpis.insert(END, 'Oznacite tocke po vrstnm redu:\n')
        self.izpis.insert(END, '#1 #2 #3\n')
        self.izpis.insert(END, '\n')
        self.izpis.insert(END, '#8    #4\n')
        self.izpis.insert(END, '\n')
        self.izpis.insert(END, '#7 #6 #5\n')

    def potrdi_porovnava(self):
        izpis_tock()
        self.izpis.insert(END, 'Izbrali ste tocke:\n')
        self.izpis.insert(END, array_coords)
        self.izpis.insert(END, '\n')
        porovnava()
        self.izpis.insert(END, 'Izvedli ste porovnavo\n')

    def potrdi_maska(self):
        razgradnja_slike_ai()
        self.izpis.insert(END, 'Izvedli ste razgradnjo slike: \n')
        self.izpis.insert(END, 'Cebele pokrivajo:')
        self.izpis.insert(END, pokritost_cebel_procent)
        self.izpis.insert(END, ' %\n')
        izracun_st_cebel()
        self.izpis.insert(END, 'Na sliki je:')
        self.izpis.insert(END, st_cebel)
        self.izpis.insert(END, 'cebel:\n')

    def rotate(self):
        st_rotacij = self.vnos_st_rotacij.get()
        st_rotacij = int(st_rotacij)
        rotation(st_rotacij)
        self.izpis.insert(END, 'Izvedli rotacijo za: ')
        self.izpis.insert(END, st_rotacij)
        self.izpis.insert(END, ' stopinj\n')

    def potrdi_quit(self):
        _quit()


okno = Tk()
my_gui = CebeleGUI(okno)
okno.mainloop()