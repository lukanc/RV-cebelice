import numpy as np
from PIL import Image as im
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy.interpolate import interpn
import cv2 as cv


##############################################################################################################################
def showImage(iImage, iTitle=''):
    plt.figure()
    plt.imshow(iImage, cmap='gray')
    plt.suptitle(iTitle)
    plt.xlabel('x')
    plt.ylabel('y')

##############################################################################################################################   
def loadImage(iPath):
    oImage = np.array(im.open(iPath))
    return oImage

##############################################################################################################################
def loadImageRaw(iPath, iSize, iFormat):
    oImage = np.fromfile(iPath, dtype= iFormat)
    oImage = oImage.reshape(iSize)
    
    return oImage

##############################################################################################################################
def saveImageRaw(iImage, iPath, iFormat):
    # dopolni
    iImage = iImage.astype(iFormat)
    iImage.tofile(iPath)

    
##############################################################################################################################   
def scaleImage( iImage, iSlopeA, iIntersectionB ):
    iImageType = iImage.dtype
    iImage = np.array(iImage, dtype = "float")
    oImage = iImage * iSlopeA + iIntersectionB
    
    if iImageType.kind in ('u', 'i'):
        oImage[oImage>np.iinfo(iImageType).max] = np.iinfo(iImageType).max
        oImage[oImage<np.iinfo(iImageType).min] = np.iinfo(iImageType).min
    return np.array(oImage, dtype=iImageType)

##############################################################################################################################
def windowImage( iImage, iCenter, iWidth ): # preslikamo sliko od iCenter +- iWidht po sivinski premisi (0-255)
    iImageType = iImage.dtype
    if np.iinfo(iImageType).kind in ('u', 'i'):
        iRange = np.iinfo(iImageType).max - np.iinfo(iImageType).min
    else:
        iRange = np.max(iImage) - np.min(iImage)
    iSlopeA = iRange/float(iWidth)
    intersectionB = -iSlopeA * (float(iCenter) - iWidth/2.0)
    
    return scaleImage(iImage, iSlopeA, intersectionB)


##############################################################################################################################
def thresholdImage(iImage, iThreshold ):
    return 255*np.array(iImage>iThreshold, dtype = 'uint8')


##############################################################################################################################
def gammaImage(iImage, iGamma):
    iImageType = iImage.dtype
    iImage = np.array(iImage, dtype = 'float')
    if np.iinfo(iImageType).kind in ('u', 'i'):
        iRange = np.iinfo(iImageType).max - np.iinfo(iImageType).min
        iMaxValue = np.iinfo(iImageType).max
        iMinValue = np.iinfo(iImageType).min
    else:
        iRange = np.max(iImage) - np.min(iImage)
        iMaxValue = np.max(iImage)
        iMinValue = np.min(iImage)
    
    iImage = (iImage - iMinValue)/iRange
    oImage = iImage ** iGamma
    oImage = float(iRange) * oImage + iMinValue
    
    if np.iinfo(iImageType).kind in ('u', 'i'):
        oImage[oImage > iMaxValue] = iMaxValue
        oImage[oImage < iMinValue] = iMinValue
    
    return np.array(oImage, dtype = iImageType)


##############################################################################################################################
def convertImageColorSpace(iImage, iConversionType): 
    iImage = np.array(iImage, dtype ='float')
    if iConversionType =='RGBtoHSV':
        r,g,b = iImage[:,:,0], iImage[:,:,1], iImage[:,:,2]
        r,g,b = r/255.0, g/255.0, b/255.0
        h,s,v = np.zeros_like(r), np.zeros_like(r), np.zeros_like(r)
        cMax = np.maximum(r, np.maximum(g,b))
        cMin =np.minimum(r, np.minimum(g,b))
        delta = cMax - cMin + 1e-7
        
        h[cMax == r] = 60.0 * (((g[cMax == r] - b[cMax == r]) / delta[cMax == r]) %6)
        h[cMax == g] = 60.0 * (((b[cMax == g] - r[cMax == g]) / delta[cMax == g]) +2)
        h[cMax == b] = 60.0 * (((r[cMax == b] - g[cMax == b]) / delta[cMax == b]) +4)
        
        s = delta / cMax
        v = cMax
        
        oImage = np.zeros_like(iImage)
        oImage[:, :, 0] = h
        oImage[:, :, 1] = s
        oImage[:, :, 2] = v
        return oImage
    
    if iConversionType =='HSVtoRGB':
        H = iImage[:, :, 0]
        S = iImage[:, :, 1]
        V = iImage[:, :, 2]
        C = np.clip(S, 0, 1) * V
        X = C * (1 - abs(((H/60)%2)-1))
        m = V - C
        x, y, z = iImage.shape
        r, g, b = np.zeros_like(H), np.zeros_like(H), np.zeros_like(H)
        for i in range(x):
            for j in range(y):
                if(0<=H[i,j]<60):
                    r[i,j] = C[i,j]
                    g[i,j] = X[i,j]
                    b[i,j] = 0

                if(60<=H[i,j]<120):
                    r[i,j] = X[i,j]
                    g[i,j] = C[i,j]
                    b[i,j] = 0

                if(120<=H[i,j]<180):
                    r[i,j] = 0
                    g[i,j] = C[i,j]
                    b[i,j] = X[i,j]

                if(180<=H[i,j]<240):
                    r[i,j] = 0
                    g[i,j] = X[i,j]
                    b[i,j] = C[i,j]

                if(240<=H[i,j]<300):
                    r[i,j] = X[i,j]
                    g[i,j] = 0
                    b[i,j] = C[i,j]

                if(300<=H[i,j]<360):
                    r[i,j] = C[i,j]
                    g[i,j] = 0
                    b[i,j] = X[i,j]           
        oImage = np.zeros_like(iImage)
        oImage[:,:,0] = r + m
        oImage[:,:,1] = g + m
        oImage[:,:,2] = b + m
        return oImage

    
##############################################################################################################################

def makeNegativeImage(iImage):
    iImageType = iImage.dtype
    iImage = np.array(iImage, dtype = "float")
    oImage = iImage * (-1) + 255
    
    if iImageType.kind in ('u', 'i'):
        oImage[oImage>np.iinfo(iImageType).max] = np.iinfo(iImageType).max
        oImage[oImage<np.iinfo(iImageType).min] = np.iinfo(iImageType).min
    
    negative_image = np.array(oImage, dtype=iImageType)
    return negative_image;

    
##############################################################################################################################


import sys
sys.path.append("/home/tilengeni/")
import knjiznica
import numpy as np
from PIL import Image as im
# print(iImage)

def calculate_threshold50(iImage):
    histogram = np.histogram(iImage, 100)
    Cvsota = np.sum(histogram[0])
    index = histogram[0].size - 1
    vsota = 0
#     print(histogram)
    while (vsota < Cvsota/2):
#         print(index, vsota, Cvsota)
        vsota = vsota + histogram[0][index]
        index = index - 1
    iIndex = index + 1
    t = histogram[1][iIndex]
#     print(vsota)
    return t
    
##############################################################################################################################


def discreteConvolution2D( iImage, iKernel ):
    """Diskretna 2D konvolucija slike s poljubnim jedrom"""    
    # pretvori vhodne spremenljivke v np polje in
    # inicializiraj izhodno np polje
    iImage = np.asarray( iImage )
    iKernel = np.asarray( iKernel )
    oImage = np.zeros_like( iImage, dtype='float' )
    # preberi velikost slike in jedra
    dy, dx = iImage.shape
    dv, du = iKernel.shape
    # izracunaj konvolucijo
    for y in range( dy ):
        for x in range( dx ):
            for v in range( dv ):
                for u in range( du ):
                    tx = int(x - u + np.floor(du/2))
                    ty = int(y - v + np.floor(dv/2))
                    if tx>=0 and tx<dx and ty>=0 and ty<dy:
                        oImage[y, x] = oImage[y, x] + \
                            float(iImage[ty, tx]) * float(iKernel[v, u])
    if iImage.dtype.kind in ('u','i'):
        oImage  = np.clip(oImage, np.iinfo(iImage.dtype).min,np.iinfo(iImage.dtype).max)
    return np.array( oImage, dtype=iImage.dtype )
    
##############################################################################################################################



def discreteConvolutionColorImage( iImage, iKernel, imode ):
    # YOUR CODE HERE 
    oImage = np.zeros_like( iImage)
    for z in range(3):
        oImage[:,:,z]=scipy.ndimage.convolve(iImage[:,:,z], iKernel, mode=imode)
    return oImage


##############################################################################################################################


def decimateImage2D(iImage, iLevel):
    """Funkcija za piramidno decimacijo"""  
    print('Decimacija pri iLevel = ', iLevel)
    # pretvori vhodne spremenljivke v np polje
    oImage = np.asarray( iImage )
    oImageType = iImage.dtype
    # gaussovo jedro za glajenje
    iKernel = np.array( ((1/16,1/8,1/16),(1/8,1/4,1/8),(1/16,1/8,1/16)) )
    # hitrejsa verzija glajenja 
    for z in range(3):
        oImage[:,:,z]=scipy.ndimage.convolve(iImage[:,:,z], iKernel, mode='nearest')  
        # decimacija s faktorjem 2
        oImage = oImage[::2,::2]
        # vrni sliko oz. nadaljuj po piramidi
        if iLevel <= 1:
            return np.array( oImage, dtype=oImageType )
        else:
            return decimateImage2D( oImage, iLevel-1 )
        

##############################################################################################################################

def discreteGaussian2D( iSigma ):
    # YOUR CODE HERE
    U= int(np.ceil(2 * (3*iSigma) + 1))
    if U%2 == 0:
        U += 1
#     print('velikost = ', U)
    oKernel = np.zeros([U,U])
    print('oKernel je dimenzije=', oKernel.shape)
    U1 = np.floor(U/2)
#     print('U1=', U1)
    for i in range(U):
        for j in range(U):
            k = ((2*np.pi)**(-1))*(iSigma**(-2)) * np.exp(-(((i-U1)**2) + ((j-U1)**2)) / (2 * (iSigma**2)))
            oKernel[i ,j] = k
            
    oKernel = oKernel/np.sum(oKernel)
    return oKernel


##############################################################################################################################



def alignICP(iPtsRef, iPtsMov, iEps=1e-6, iMaxIter=50, plotProgress=False):
    """Postopek iterativno najblizje tocke"""
    # inicializiraj izhodne parametre
    curMat = []; oErr = []; iCurIter = 0
    if plotProgress:
        iPtsMov0 = np.matrix(iPtsMov)
        fig = plt.figure()
        ax = fig.add_subplot(111)
    # zacni iterativni postopek
    while True:
        # poisci korespondencne pare tock
        iPtsRef_t, iPtsMov_t = findCorrespondingPoints(iPtsRef, iPtsMov)
        # doloci afino aproksimacijsko preslikavo
        oMat2D = mapAffineApprox2D(iPtsRef_t, iPtsMov_t)
        # posodobi premicne tocke
        iPtsMov = np.dot(addHomCoord2D(iPtsMov), oMat2D.transpose())
        # izracunaj napako
        curMat.append(oMat2D)
        oErr.append(np.sqrt(np.sum((iPtsRef_t[:,:2]- iPtsMov_t[:,:2])**2)))
        iCurIter = iCurIter + 1 
        # preveri kontrolne parametre        
        dMat = np.abs(oMat2D - transAffine2D())
        if iCurIter>iMaxIter or np.all(dMat<iEps):
            break

    # doloci kompozitum preslikav
    oMat2D = transAffine2D()   
    for i in range(len(curMat)):
        
        if plotProgress:
            iPtsMov_t = np.dot(addHomCoord2D(iPtsMov0), oMat2D.transpose())
            ax.clear()   
            ax.plot(iPtsRef[:,0], iPtsRef[:,1], 'ob')   
            ax.plot(iPtsMov_t[:,0], iPtsMov_t[:,1], 'om')
            fig.canvas.draw()
            plt.pause(1)
        
        oMat2D = np.dot(curMat[i], oMat2D)
    return oMat2D, oErr


##############################################################################################################################



def findCorrespondingPoints(iPtsRef, iPtsMov):
    """Poisci korespondence kot najblizje tocke"""
    # inicializiraj polje indeksov
    iPtsMov = np.array(iPtsMov)
    iPtsRef = np.array(iPtsRef)
    
    idxPair = -np.ones((iPtsRef.shape[0], 1), dtype='int32')
    idxDist = np.ones((iPtsRef.shape[0], iPtsMov.shape[0]))
    for i in range(iPtsRef.shape[0]):
        for j in range(iPtsMov.shape[0]):
            idxDist[i,j] = np.sum((iPtsRef[i,:2] - iPtsMov[j,:2])**2)
    # doloci bijektivno preslikavo
    while not np.all(idxDist==np.inf):            
        i, j = np.where(idxDist == np.min(idxDist))
        idxPair[i[0]] = j[0]
        idxDist[i[0],:] = np.inf
        idxDist[:,j[0]] = np.inf            
    # doloci pare tock
    idxValid, idxNotValid = np.where(idxPair>=0)
    idxValid = np.array( idxValid )               
    iPtsRef_t = iPtsRef[idxValid,:]
    iPtsMov_t = iPtsMov[idxPair[idxValid].flatten(),:]
    return iPtsRef_t, iPtsMov_t


##############################################################################################################################

def mapAffineApprox2D(iPtsRef, iPtsMov):
    """Afina aproksimacijska poravnava"""
    iPtsRef = np.matrix(iPtsRef) 
    iPtsMov = np.matrix(iPtsMov) 
    # po potrebi dodaj homogeno koordinato
    iPtsRef = addHomCoord2D(iPtsRef)
    iPtsMov = addHomCoord2D(iPtsMov)
    # afina aproksimacija (s psevdoinverzom)
    iPtsRef = iPtsRef.transpose()
    iPtsMov = iPtsMov.transpose()
#     print(iPtsRef.shape)
#     print(iPtsMov.shape)
    # psevdoinverz
    oMat2D = np.dot(iPtsRef, np.linalg.pinv(iPtsMov))        
    # psevdoinverz na dolgo in siroko:
    #oMat2D = iPtsRef * iPtsMov.transpose() * \
    #np.linalg.inv( iPtsMov * iPtsMov.transpose() )               
    return oMat2D


##############################################################################################################################

def transAffine2D(iScale=(1, 1), iTrans=(0, 0), iRot=0, iShear=(0, 0)):
    """Funkcija za poljubno 2D afino preslikavo"""
    iRot = iRot * np.pi / 180
    oMatScale = np.array( ((iScale[0],0,0),(0,iScale[1],0),(0,0,1)) )
    oMatTrans = np.array( ((1,0,iTrans[0]),(0,1,iTrans[1]),(0,0,1)) )
    oMatRot = np.array(((np.cos(iRot),-np.sin(iRot),0),\
                        (np.sin(iRot),np.cos(iRot),0),
                        (0,0,1)))
    oMatShear = np.array( ((1,iShear[0],0),(iShear[1],1,0),(0,0,1)) )
    # ustvari izhodno matriko
    oMat2D = np.dot(oMatTrans, np.dot(oMatShear, np.dot(oMatRot, oMatScale)))
    return oMat2D


##############################################################################################################################

def addHomCoord2D(iPts):
    if iPts.shape[-1] == 3:
        return iPts
    iPts = np.hstack((iPts, np.ones((iPts.shape[0], 1))))
    return iPts

##############################################################################################################################


def addHomCoord3D(iPts):
    if iPts.shape[-1] == 4:
        return iPts
    iPts = np.hstack((iPts, np.ones((iPts.shape[0], 1))))
    return iPts

##############################################################################################################################

def interpolate0Image2D( iImage, iCoorX, iCoorY ):
    """Funkcija za interpolacijo nictega reda"""
    # pretvori vhodne spremenljivke v np polje
    iImage = np.asarray( iImage )    
    iCoorX = np.asarray( iCoorX, dtype='float')
    iCoorY = np.asarray( iCoorY, dtype='float')   
    # preberi velikost slike in jedra
    dy, dx = iImage.shape
    # ustvari 2d polje koordinat iz 1d vhodnih koordinat (!!!)
    if np.size(iCoorX) != np.size(iCoorY):
        print('Stevilo X in Y koordinat je razlicno!')      
        iCoorX, iCoorY = np.meshgrid(iCoorX, iCoorY, indexing='xy')
    # zaokrozi na najblizjo celostevilsko vrednost (predstavlja indeks!)
    oShape = iCoorX.shape   
    iCoorX = np.floor(iCoorX.flatten()).astype('int')
    iCoorY = np.floor(iCoorY.flatten()).astype('int')
    # ustvari izhodno polje    
    oImage = np.zeros(iCoorX.shape, dtype=iImage.dtype )
    print(iCoorX.shape)
    print(iCoorY.shape)
    # priredi vrednosti    
    for idx in range(oImage.size):
        tx = iCoorX[idx]
        ty = iCoorY[idx]
        if tx>=0 and tx<dx and ty>=0 and ty<dy:
            oImage[idx] = iImage[ty, tx]
    # vrni izhodno sliko
    return np.reshape( oImage, oShape )


##############################################################################################################################

def interpolateColorImage( iImage, iCoorX, iCoorY, method2 ):
    oImage = np.asarray( iImage )
    dy, dx, dz = iImage.shape
    oImage = interpn((np.arange(dy),np.arange(dx)),iImage,np.dstack((iCoorY,iCoorX)),method= method2)
    oImage = oImage.astype(int)
    return oImage

##############################################################################################################################



def drawLine(iImage, iValue, x1, y1, x2, y2):
    ''' Narisi digitalno daljico v sliko

        Parameters
        ----------
        iImage : numpy.ndarray
            Vhodna slika
        iValue : tuple, int
            Vrednost za vrisavanje (barva daljice).
            Uporabi tuple treh elementov za barvno sliko in int za sivinsko sliko
        x1 : int
            Začetna x koordinata daljice
        y1 : int
            Začetna y koordinata daljice
        x2 : int
            Končna x koordinata daljice
        y2 : int
            Končna y koordinata daljice
    '''    
    
    oImage = iImage    
    
    if iImage.ndim == 3:
        assert type(iValue) == tuple, 'Za barvno sliko bi paramter iValue moral biti tuple treh elementov'
        for rgb in range(3):
            drawLine(iImage[rgb,:,:], iValue[rgb], x1, y1, x2, y2)
    
    elif iImage.ndim == 2:
        assert type(iValue) == int, 'Za sivinsko sliko bi paramter iValue moral biti int'
    
        dx = np.abs(x2 - x1)
        dy = np.abs(y2 - y1)
        if x1 < x2:
            sx = 1
        else:
            sx = -1
        if y1 < y2:
            sy = 1
        else:
            sy = -1
        napaka = dx - dy
     
        x = x1
        y = y1
        
        while True:
            oImage[y-1, x-1] = iValue
            if x == x2 and y == y2:
                break
            e2 = 2*napaka
            if e2 > -dy:
                napaka = napaka - dy
                x = x + sx
            if e2 < dx:
                napaka = napaka + dx
                y = y + sy
    
    return oImage

##############################################################################################################################


def geomCalibImageRGB( iPar, iImage, iCoorX, iCoorY, step = 1 ):
    """Funkcija za normalizacijo slike po geometrijski kalibraciji"""
    oCoorU, oCoorV = geomCalibTrans( iPar, iCoorX, iCoorY )
#     print(iImage.shape)
    dy, dx, dz = iImage.shape
#     print('y:',dy, 'X:', dx)
    oImage = interpn((np.arange(dy), np.arange(dx)), 
                            iImage, 
                            (oCoorV, oCoorU), 
                            method = "linear", bounds_error=False)
    print(oImage.shape)
    iCoorX, iCoorY = np.meshgrid(np.arange(0,2260,1/step), np.arange(0,3840,1/step), indexing='xy')
    oImage = np.nan_to_num(oImage)
#     print(oImage)
    oImage = knjiznica.interpolateColorImage(oImage, iCoorX, iCoorY, 'nearest')
    return oImage

##############################################################################################################################

def geomCalibImage( iPar, iImage, iCoorX, iCoorY, step = 1 ):
    """Funkcija za normalizacijo slike po geometrijski kalibraciji"""
    oCoorU, oCoorV = geomCalibTrans( iPar, iCoorX, iCoorY )
    print(iImage.shape)
    dy, dx = iImage.shape
    print('y:',dy, 'X:', dx)
    oImage = interpn((np.arange(dy), np.arange(dx)), 
                            iImage, 
                            (oCoorV, oCoorU), 
                            method = "linear", bounds_error=False)
    print(oImage.shape)
    iCoorX, iCoorY = np.meshgrid(np.arange(0,280,1/step), np.arange(0,200,1/step), indexing='xy')
    
    oImage = knjiznica.interpolate0Image2D( oImage, iCoorX, iCoorY )
    return oImage


##############################################################################################################################

def geomCalibTrans( iPar, iCoorX, iCoorY ):
    """Funkcija za preslikavo tock (projektivna+radialne distorzije)"""
    iParProj = iPar[0:8]
    iParRad = iPar[8:]
    # preslikava v prostor slike
    iCoorUt, iCoorVt = transProjective2D( iParProj, iCoorX, iCoorY )
    # korekcija radialnih distorzij
    iCoorUt, iCoorVt = transRadial(iParRad[2:], iParRad[0], iParRad[1], 
                                   iCoorUt, iCoorVt)
    # vrni preslikane tocke
    return iCoorUt, iCoorVt




##############################################################################################################################

def geomCalibErr( iPar, iCoorU, iCoorV, iCoorX, iCoorY):
    """Funkcija za izracun kalibracijske napake"""
    # preslikaj tocke z dano preslikavo
    iCoorUt, iCoorVt = geomCalibTrans( iPar, iCoorX, iCoorY )
    #izracun napake poravnave
    oErr2 = np.mean( (iCoorU-iCoorUt)**2 + (iCoorV-iCoorVt)**2 )
    # vrni vrednost napake
    return oErr2



##############################################################################################################################

# Pretvori v sivinsko sliko
def colorToGray(iImage):
    dtype = iImage.dtype
    r = iImage[:, :, 0].astype('float')
    g = iImage[:, :, 1].astype('float')
    b = iImage[:, :, 2].astype('float')

    return (r * 0.299 + g * 0.587 + b * 0.114).astype(dtype)


##############################################################################################################################


def transRadial(iK, iUc, iVc, iCoorU, iCoorV):
    """Funkcija za preslikavo z Brownovim modelom distorzij"""
    # preveri vhodne podatke
    iK = np.array(iK)
    iCoorU = np.array(iCoorU)
    iCoorV = np.array(iCoorV)
    if np.size(iCoorU) != np.size(iCoorV):
        print('Stevilo U in V koordinat je razlicno!')
        # odstej koodinate centra
    oCoorUd = iCoorU - iUc;
    oCoorVd = iCoorV - iVc
    # pripravi izhodne koordinate
    sUd = np.max(np.abs(oCoorUd))
    sVd = np.max(np.abs(oCoorVd))
    oCoorUd = oCoorUd / sUd
    oCoorVd = oCoorVd / sVd
    # preracunaj radialna popacenja
    R2 = oCoorUd ** 2.0 + oCoorVd ** 2.0
    iK = iK.flatten()
    oCoorRd = np.ones_like(oCoorUd)
    for i in range(iK.size):
        oCoorRd = oCoorRd + iK[i] * (R2 ** (i + 1))
    # izracunaj izhodne koordinate
    oCoorUd = oCoorUd * oCoorRd * sUd + iUc
    oCoorVd = oCoorVd * oCoorRd * sVd + iVc
    return oCoorUd, oCoorVd



##############################################################################################################################


def transProjective2D(iPar, iCoorX, iCoorY):
    """Funkcija za projektivno preslikavo"""
    # preveri vhodne podatke
    iPar = np.asarray(iPar)
    iCoorX = np.asarray(iCoorX)
    iCoorY = np.asarray(iCoorY)
    if np.size(iCoorX) != np.size(iCoorY):
        print('Stevilo X in Y koordinat je razlicno!')
        # izvedi projektivno preslikavo
    oDenom = iPar[6] * iCoorX + iPar[7] * iCoorY + 1
    oCoorU = iPar[0] * iCoorX + iPar[1] * iCoorY + iPar[2]
    oCoorV = iPar[3] * iCoorX + iPar[4] * iCoorY + iPar[5]
    # vrni preslikane tocke
    return oCoorU / oDenom, oCoorV / oDenom



##############################################################################################################################

def enhanceLinear(iImage, iSigma, iBeta):
    Obcutljivost = 5
    imgG = cv.cvtColor(iImage, cv.COLOR_RGB2GRAY)
    blur = cv.GaussianBlur(imgG, (3, 3), 0)

    test = cv.cornerEigenValsAndVecs(blur, iSigma, 3)
    test2 = test[:, :, 0:2]

    Lambda1 = test[:, :, 0]
    Lambda2 = test[:, :, 1]
    #     print('Lambda1', Lambda1.shape)
    #     print('Lambda2', Lambda2.shape)

    QLA = (Lambda1 - Lambda2) / (Lambda1 + Lambda2 - iBeta)
    oQLA = np.int8(-(QLA * 255) * Obcutljivost)

    return oQLA


##############################################################################################################################

def razgradnja(iImage):

    import random
    import numpy as np

    from keras.models import Model, load_model
    from keras import backend as K
    import tensorflow as tf


    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    seed = 42
    random.seed = seed
    np.random.seed = seed


    ### Definicija kriterijskih funkcij
    def mean_iou(y_true, y_pred):
        prec = []
        for t in np.arange(0.5, 1.0, 0.05):
            y_pred_ = tf.to_int32(y_pred > t)
            score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
            K.get_session().run(tf.local_variables_initializer())
            with tf.control_dependencies([up_opt]):
                score = tf.identity(score)
            prec.append(score)
        return K.mean(K.stack(prec), axis=0)

    def iou_coef(y_true, y_pred, smooth=1):
        """
        IoU = (|X &amp; Y|)/ (|X or Y|)
        """
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
        return (intersection + smooth) / ( union + smooth)

    def iou_coef_loss(y_true, y_pred):
        return -iou_coef(y_true, y_pred)

    def dice_coef(y_true, y_pred):
        """
        DSC = (2*|X &amp; Y|)/ (|X| + |Y|)
        """
        smooth = 1
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def dice_coef_loss(y_true, y_pred):
        return 1-dice_coef(y_true, y_pred)


    ### Konstante in parametri

    ### 3.1 Naloži MRI podatke in loči med učne in testne

    slike_sub_array = dict()
    cebele_data_slike = []


    from os.path import join

    for i in range(10):
        for j in range(10):
            slike_sub_array[i,j] = np.array(iImage[i * 384:(i + 1) * 384, j * 226:(j + 1) * 226])
            cebele_data_slike.append((slike_sub_array[i,j]))

    X_test = np.asarray(cebele_data_slike)

    ## PADING
    X_test = np.pad(X_test, [(0,0),(0,0),(0,30),(0,0)], mode='constant', constant_values=0)


    # print('Velikost 4D polja za testiranje: {}'.format(X_test.shape))

    ### 3.3 Vrednotenje razgradnje

    # naloži model
    model = load_model(join('models', 'model-cebele-unet-seg-1.h5'),
                       custom_objects={'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss})


    Rezultati = []

    # opravi razgradnjo na učni in testni zbirki
    preds_test = model.predict(X_test, verbose=1)

    # dobljene vrednosti
    preds_test_t = (preds_test > 0.3).astype(np.uint8)
    Rezultati.append(preds_test_t)
    Rezultati = np.asarray(Rezultati)
    # print('Rezultati shape:', Rezultati.shape)

    # print('stevilo pixlov Unet', np.count_nonzero(preds_test_t))
    return(preds_test_t)

##############################################################################################################################




##############################################################################################################################
