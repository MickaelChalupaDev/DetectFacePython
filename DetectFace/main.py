import cv2
import mediapipe as mp
import time

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
pTime = 0

#cap = cv2.VideoCapture(0)
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

frameWidth = 1920
frameHeight = 1080
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = faceMesh.process(imgRGB) # construit le maillage
    if result.multi_face_landmarks:
        for faceLms in result.multi_face_landmarks: # pour chaque maillage (de chaque visage trouvé)
            # Dessin du visage
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)

    cTime = time.time() # (mili) seconde actuelle
    fps = 1 / (cTime - pTime) # fps = 1 (seconde actuelle - seconde précédente)
    pTime = cTime #met à jour la seconde précédente à chaque affichage d'image
    cv2.imshow("Image", img) # montre la vidéo
    if cv2.waitKey(1) == 27: # touche 27 = touche ECHAP
        break # sortie de la boucle et fin du programme

cap.release()