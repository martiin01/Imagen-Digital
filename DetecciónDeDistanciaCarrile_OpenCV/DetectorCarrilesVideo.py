import cv2
import numpy as np

def mostrar_lineas(img, lineas):
    img_lineas = np.zeros_like(img)
    if lineas is not None:
        for linea in lineas:
            x1, y1, x2, y2 = linea[0]
            cv2.line(img_lineas, (x1, y1), (x2, y2), (0, 255, 0), 10)
    img_combinar = cv2.addWeighted(img, 0.8, img_lineas, 1, 1)
    return img_combinar

def calcular_desviacion(img, lineas):
    ancho = img.shape[1]
    if lineas is not None:
        x_posiciones = []
        for linea in lineas:
            x1, y1, x2, y2 = linea[0]
            x_posiciones.extend([x1, x2])
        x_media = np.mean(x_posiciones)
        centro_imagen = ancho / 2
        desviacion = x_media - centro_imagen
    else:
        desviacion = 0
    return desviacion

# Capturar el vídeo
cap = cv2.VideoCapture('carretera4.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocesamiento
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gris_suavizado = cv2.GaussianBlur(gris, (5, 5), 0)
    bordes = cv2.Canny(gris_suavizado, 50, 150)

    # Definir región de interés (ROI)
    altura = frame.shape[0]
    ancho = frame.shape[1]
    poligono = np.array([[
        (0, altura),
        (ancho, altura),
        (ancho, int(altura * 0.6)),
        (0, int(altura * 0.6))
    ]])
    mascara = np.zeros_like(bordes)
    cv2.fillPoly(mascara, poligono, 255)
    recorte = cv2.bitwise_and(bordes, mascara)

    # Detectar líneas con la Transformada de Hough
    lineas = cv2.HoughLinesP(recorte, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=150)

    # Mostrar líneas detectadas
    img_resultado = mostrar_lineas(frame, lineas)

    # Calcular desviación
    desviacion = calcular_desviacion(frame, lineas)
    print(f"Desviación: {desviacion}")

    # Mostrar resultado
    cv2.imshow('Detección de Carriles', img_resultado)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()