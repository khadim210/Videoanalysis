# Author: Ahmath B. MBACKE
# # Date: 2023-10-01
# Description: Ce code analyse les émotions dans une vidéo et génère des courbes d'évolution des émotions au fil du temps.
# Il peut être utilisé pour des applications de reconnaissance émotionnelle dans des vidéos, notament les enregistrements de réunions ou webinaires.

import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt

# === Paramètres ===
video_path = "videoMashup.mp4"
output_video_path = "video_avec_faces.mp4"
output_plot_paths = {
    'Joie': 'courbe_joie.png',
    'Tristesse': 'courbe_tristesse.png',
    'Colère': 'courbe_colere.png',
    'Surprise': 'courbe_surprise.png'
}

# Initialisation vidéo
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Vidéo annotée en sortie
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Stockage émotions
time_stamps = []
joy_series = []
sad_series = []
angry_series = []
surprise_series = []

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % fps == 0:
        try:
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

            if not isinstance(results, list):
                results = [results]

            joy_sum, sad_sum, angry_sum, surprise_sum = 0, 0, 0, 0

            for face in results:
                emotion = face['emotion']
                region = face['region']
                x, y, w, h = region['x'], region['y'], region['w'], region['h']

                if w > 0 and h > 0:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    top_emotion = max(emotion, key=emotion.get)
                    confidence = int(emotion[top_emotion])
                    label = f"{top_emotion}: {confidence}%"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                joy_sum += emotion.get('happy', 0)
                sad_sum += emotion.get('sad', 0)
                angry_sum += emotion.get('angry', 0)
                surprise_sum += emotion.get('surprise', 0)

            time_stamps.append(frame_id / fps)
            joy_series.append(joy_sum)
            sad_series.append(sad_sum)
            angry_series.append(angry_sum)
            surprise_series.append(surprise_sum)

        except Exception as e:
            print(f"Erreur à la frame {frame_id}: {e}")

    # Affichage temps réel
    cv2.imshow("Analyse des émotions", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("⏹️ Interruption par l'utilisateur.")
        break

    out.write(frame)
    frame_id += 1

cap.release()
out.release()
cv2.destroyAllWindows()

# === Sauvegarde des courbes individuelles ===
def save_curve(times, values, label, filename):
    plt.figure(figsize=(8, 4))
    plt.plot(times, values, label=label)
    plt.xlabel('Temps (s)')
    plt.ylabel('Niveau cumulé')
    plt.title(f'Évolution de l\'émotion : {label}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"✅ Courbe {label} enregistrée sous : {filename}")

save_curve(time_stamps, joy_series, 'Joie', output_plot_paths['Joie'])
save_curve(time_stamps, sad_series, 'Tristesse', output_plot_paths['Tristesse'])
save_curve(time_stamps, angry_series, 'Colère', output_plot_paths['Colère'])
save_curve(time_stamps, surprise_series, 'Surprise', output_plot_paths['Surprise'])
