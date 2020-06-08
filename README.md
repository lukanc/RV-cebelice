# RV-cebelice
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![Generic badge](https://img.shields.io/badge/python-3-green.svg?style=for-the-badge&logo=appveyor)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/FE-LST-red.svg?style=for-the-badge&logo=appveyor)](https://shields.io/)


Projekt pri predmetu Robotski vid

~~v razvoju~~, RAZVITO

## O projektu
Projekt je nastal kot seminar v poletnem semestru 1. letnika magisterskega študija smer robotika pri predmetu Robotski vid. 
Seminarska naloga je nastala v sodelovanju s kmetijskim inštitutom slovenije

Mentor: As. dr. Žiga Špiclin

Avtor slik : doc. dr. Janez Prešern


## Navodila za upravljanje s programom

1. Zaženemo datoteko GUI_cebele
2. Ob zagonu se nam prikaže gui.
3. Nato izberemo željeno sliko,
4. Po potrebi poravnamo tako da napišemo stevilo stopinj za katerih želimo zarotirati sliko in pritisnemo tipko Zarotiraj sliko
5. Pritisnemo tipko "Oznaci tocke" in označimo notranji rob okvirja satovja.
Ko smo označili 8 točk v vrstnem redu kot piše v programu:

#1 #2 #3

#8........#4

#7 #6 #5

![alt text](https://github.com/lukanc/RV-cebelice/blob/master/ozna%C4%8Devanje_za_geom_kalib.png?raw=true)

6. Pritisnemo tipko izvedi poravnavo.
7. Po poravnavi pritisnemo tipko Izracun st cebel
8. Program nam vrne število čebel na poravnani sliki.

---
### Avtorja

Tilen Kočevar tk2614@student.uni-lj.si

Luka Kresnik lk1712@student.uni-lj.si

---

## Requirements 

* cv2
* scipy
* matplotlib
* PIL
* Numpy
* tkinter
* keras
* Tensorflow
