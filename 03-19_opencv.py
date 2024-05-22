import cv2
import matplotlib.pyplot as plt
from PIL import Image
import imutils

cv2.__version__


# WCZYTANIE I WYSWIETLENIE OBRAZU CV2
img = cv2.imread(filename=r'C:\Users\kpas9\Pictures\XC60 II\2018-Volvo-XC60-1.jpg')
# Convert BGR to RGB (OpenCV loads images as BGR by default)
image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.show()


# WCZYTANIE I WYSWIETLENIE OBRAZU PILLOW
image = Image.open(r'C:\Users\kpas9\Pictures\XC60 II\2018-Volvo-XC60-1.jpg')
plt.imshow(image)
plt.show()


img

# OBRAZ 3 KANAŁOWY (960, 1280, 3), WYSOKOSC 960, SZERORKOSC 1280, 3 KOLORY(RGB)
img.shape

# UTWORZENIE KOPII OBRAZU
cv2.imwrite(filename='copy.png', img=img)

# WCZYTANIE OBRAZU W SKALI SZAROSCI
img = cv2.imread(filename='copy.png', flags=cv2.IMREAD_GRAYSCALE)
img

plt.imshow(img)
plt.show()


# TAKI OBRAZ POSIADA JUŻ TYLKO WYS, SZER (960,1280), NIE POSIADA ROZMIARU WSKAZUJACEGO NA KOLORY - RGB
img.shape










# WYCINEK OBRAZU LEWY GORNY ROG, 0 ZACZYNA SIE W TYM ROGU, NAJPIERW PODAJEMY WYSOKOSC, POZNIEJ SZEROKOSC WYCINKA
left_top_square = img[:200, :400]
plt.imshow(left_top_square)
plt.show()

right_top_corner = img[:200, -200:]
plt.imshow(right_top_corner)
plt.show()

left_bottom_corner = img[-200:, :200]
plt.imshow(left_bottom_corner)
plt.show()

right_botton_corner = img[-200:,-200:]
plt.imshow(right_botton_corner)
plt.show()


# WYZNACZENIE PUNKTOW OKOLOSRODKOWYCH
height, width, _ = img.shape
h0 = height//2 - 100
h1 = height//2 + 100
w0 = width//2 - 100
w1 = width//2 + 100

# WYCINEK SRODKA ZDJECIA
middle = img[h0:h1, w0:w1]
plt.imshow(middle)
plt.show()







# ZMIANA ROZMIARU ZDJECIA
img_resized = imutils.resize(image=img, height=300)
plt.imshow(img_resized)
plt.show()

img_resized.shape








# DETEKCJA KRAWEDZI LAPLACIAN
plt.imshow(cv2.Laplacian(src=img, ddepth=cv2.CV_64F))
plt.show()

# DETEKCJA KRAWEDZI CANNY
plt.imshow(cv2.Canny(image=img,threshold1=200, threshold2=200))
plt.show()







# RYSOWANIE PO OBRAZIE
# KWADRAT O KATACH PO PRZEKATNEJ PT1, PT2
plt.imshow(cv2.rectangle(img=img.copy(), pt1=(75,80), pt2=(180,200), color=(0,0,255), thickness=2))
plt.imshow(cv2.rectangle(img=img.copy(), pt1=(140,250), pt2=(350,470), color=(0,0,255), thickness=2))
plt.show()

# RYSOWANIE KOLA I OKREGU
# THICKNESS < 0 DAJE KOLO
# THICKNESS > 0 DAJE OKRAG
plt.imshow(cv2.circle(img=img.copy(), center=(300,250), radius=100, color=(0,0,255), thickness=-1))
plt.imshow(cv2.circle(img=img.copy(), center=(300,250), radius=100, color=(0,0,255), thickness=4))

# DODANIE NAPISU
# org TO PUNKT STARTOWY DLA TEKSTU
plt.imshow(cv2.putText(img=img.copy(), text='OPENCV: VOLMO', org=(10,40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.1, color=(0,255,0), thickness=4))














# DETEKTOR PROSTOKATOW Z OBRAZU # 
# WYCINA ELEMENT NAJBARDZIEJ PODOBNY DO PROSTOKATA # 

image = cv2.imread('phone.jpg')
image = imutils.resize(image, height=400)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

grey_image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
plt.imshow(grey_image)

edges1 = cv2.Canny(image=grey_image, threshold1=70, threshold2=200)
plt.imshow(edges1)



# CZESTA PRAKTYKA PRZED WYKRYWANIEM KRAWEDZI JEST ROZMAZANIE OBRAZU, POMAGA TO WYTLUMIC WIDOCZNY SZUM  


# WYTLUMIENIE SZUMU:
# ksize - 
# sigmaX - 
grey_image = cv2.GaussianBlur(src=grey_image, ksize=(5,5), sigmaX=0)
plt.imshow(grey_image)


# FILTR CANNY NALOZONY NA ROZMAZANY OBRAZ MA MNIEJ SZUMU
edges = cv2.Canny(image=grey_image, threshold1=70, threshold2=200)
plt.imshow(edges)


# ZNALEZIENIE KONTUR
contours = cv2.findContours(image=edges.copy(), 
                            mode=cv2.RETR_TREE, 
                            method=cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

# 10 NAJWIEKSZYCH ELEMENTOW Z LISTY KONTUR
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

# NALOZENIE KONTUR NA ORGINALNY OBRAZEK
# KONKRETNA KONTURE WYKRYTA MOZNA ZMIENIAC PODAJAC JEJ INDEX W contours
cnt1 = cv2.drawContours(image=image.copy(), contours=[contours[0]], contourIdx=-1, color=(0,255,0), thickness=3)
plt.imshow(cnt1)



# ZNALEZIENIE PROSTOKATA DOPASOWANEGO DO ZADANEGO KONTURU


screed_contour = None
# DLA KAZDEJ WARTOSCI W CONTORUS
for contour in contours:
    # OBLICZENIE DLUGOSCI KONKRETNEGO KONTURU
    perimeter = cv2.arcLength(curve=contour, closed=True)
    # PRZYBLIZA KONTUR DO WIELOKATA 
    approx = cv2.approxPolyDP(curve=contour, epsilon=0.015 * perimeter, closed=True)
    # JESLI APPROX MA 4 KATY WYZNACZA KONTUR, ZWRACA 4 PUNKTY(WSPOLRZEDNE TYCH PUNKTOW)
    if len(approx) == 4:
        screed_contour = approx
        break


screed_contour

# WYSWIETLENIE ZNALEZIONYCH PUNKTOW
vertices = cv2.drawContours(image=image.copy(), contours=screed_contour, contourIdx=-1, color=(0,255,0), thickness=10)
plt.imshow(vertices)

# WYSWIETLENIE ZNALEZIONEGO PROSTOKATA
screen_countour = cv2.drawContours(image=image.copy(), contours=[screed_contour], contourIdx=-1, color=(0,255,0), thickness=3)
plt.imshow(screen_countour)

