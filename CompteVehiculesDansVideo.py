import cv2
from ultralytics import YOLO
import openpyxl
import sqlite3
import matplotlib.pyplot as plt

# Initialisation du modèle
model = YOLO("yolov8n.pt")

# Classes à suivre
CLASSES_VEHICULES = {"car", "truck", "bus", "motorbike"}
CLASSES_PERSONNES = {"person"}

#hash set pour les noms des classes de vehicules et des personnes
# car -> VL
# truck -> PL
# bus -> PL
# motorbike -> VL
# person -> PI

labels_hash = { "car": "VL", "truck": "PL", "bus": "PL", "motorbike": "VL", "person": "PI" }


# Sets pour les IDs uniques
ids_vehicules = set()
ids_personnes = set()

# Détection + suivi + affichage
def analyser_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erreur d'ouverture de la vidéo")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, conf=0.4, iou=0.8)[0]

        for box in results.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            track_id = int(box.id[0]) if box.id is not None else None
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label in CLASSES_VEHICULES and track_id is not None:
                ids_vehicules.add(track_id)
                color = (0, 255, 0)
            elif label in CLASSES_PERSONNES and track_id is not None:
                ids_personnes.add(track_id)
                color = (0, 0, 255)
            else:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            tt = labels_hash[label]

            cv2.putText(frame, f"{tt} ID:{track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Affichage du compteur
        cv2.putText(frame, f"Véhicules uniques: {len(ids_vehicules)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Personnes uniques: {len(ids_personnes)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Détection et suivi", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Enregistrement Excel
def enregistrer_excel(nb_vehicules, nb_personnes):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Détection"
    ws.append(["Catégorie", "Nombre détecté"])
    ws.append(["Véhicules", nb_vehicules])
    ws.append(["Personnes", nb_personnes])
    wb.save("resultats_detection.xlsx")
    print("✔️ Résultats enregistrés dans Excel")

# Enregistrement SQLite
def enregistrer_sqlite(nb_vehicules, nb_personnes):
    conn = sqlite3.connect("resultats.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detection (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vehicules INTEGER,
            personnes INTEGER
        )
    ''')
    cursor.execute('INSERT INTO detection (vehicules, personnes) VALUES (?, ?)',
                   (nb_vehicules, nb_personnes))
    conn.commit()
    conn.close()
    print("✔️ Résultats enregistrés dans SQLite")

# Graphique en barres
def afficher_graphique(nb_vehicules, nb_personnes):
    categories = ["Véhicules", "Personnes"]
    valeurs = [nb_vehicules, nb_personnes]
    plt.figure(figsize=(6, 4))
    plt.bar(categories, valeurs, color=["green", "red"])
    plt.title("Objets détectés")
    plt.ylabel("Nombre")
    plt.savefig("graphique_final.png")
    plt.show()
    print("✔️ Graphique enregistré dans graphique_final.png")

# Programme principal
def main():
    video_path = "video2.mp4"
    analyser_video(video_path)

    nb_vehicules = len(ids_vehicules)
    nb_personnes = len(ids_personnes)

    enregistrer_excel(nb_vehicules, nb_personnes)
    enregistrer_sqlite(nb_vehicules, nb_personnes)
    afficher_graphique(nb_vehicules, nb_personnes)

    print(f"\n🧾 Résumé :\n- Véhicules détectés : {nb_vehicules}\n- Personnes détectées : {nb_personnes}")

if __name__ == "__main__":
    main()
