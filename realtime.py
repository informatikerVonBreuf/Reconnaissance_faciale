import dlib
import cv2

# Charger le détecteur de visage de dlib
detector = dlib.get_frontal_face_detector()

# Ouvrir la webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Obtenir une image depuis la webcam
    ret, frame = video_capture.read()

    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détecter les visages dans l'image
    faces = detector(gray)

    # Dessiner un rectangle autour de chaque visage détecté
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Afficher l'image avec les visages détectés
    cv2.imshow('Face Detection', frame)

    # Quitter l'application en appuyant sur 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Fermer la webcam
video_capture.release()
cv2.destroyAllWindows()
