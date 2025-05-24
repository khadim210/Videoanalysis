import cv2
from ultralytics import YOLO
import openpyxl
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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



# Zones d'entrée/sortie définies manuellement
ZONES = {
    'A': (300, 20, 480, 100),       # Haut
    'B': (400, 125, 600, 500),    # Droite
    'C': (0, 100, 250, 220),     # Bas
    'D': (0, 250, 300, 500),       # Gauche
}

# Sets et dictionnaires pour suivi
ids_vehicules = set()
ids_personnes = set()
trajectoires = {}

# Fonction pour déterminer la zone d'un point
def get_zone(x, y):
    for zone_name, (x1, y1, x2, y2) in ZONES.items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            return zone_name
    return None

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

        results = model.track(frame, persist=True, conf=0.4)[0]

        for box in results.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            track_id = int(box.id[0]) if box.id is not None else None
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            zone = get_zone(center_x, center_y)

            if label in CLASSES_VEHICULES and track_id is not None:
                ids_vehicules.add(track_id)
                color = (0, 255, 0)
                if track_id not in trajectoires:
                    if zone:
                        trajectoires[track_id] = {"entree": zone, "sortie": zone}
                else:
                    if zone:
                        trajectoires[track_id]["sortie"] = zone

            elif label in CLASSES_PERSONNES and track_id is not None:
                ids_personnes.add(track_id)
                color = (0, 0, 255)
            else:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            tt = labels_hash[label]
            cv2.putText(frame, f"{tt} ID:{track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Dessiner les zones d'entrée/sortie
        for zone_name, (zx1, zy1, zx2, zy2) in ZONES.items():
            cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (255, 255, 0), 2)
            cv2.putText(frame, f"Zone {zone_name}", (zx1 + 5, zy1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

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
def enregistrer_excel(nb_vehicules, nb_personnes, matrice_trafic):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Détection"
    ws.append(["Catégorie", "Nombre détecté"])
    ws.append(["Véhicules", nb_vehicules])
    ws.append(["Personnes", nb_personnes])

    ws2 = wb.create_sheet(title="Trafic Entrée-Sortie")
    ws2.append(["Entrée", "Sortie", "Nombre de véhicules"])
    for (entree, sortie), count in matrice_trafic.items():
        ws2.append([entree, sortie, count])

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

# Heatmap de la matrice Entrée/Sortie
def afficher_matrice_trafic(matrice_trafic):
    zones = sorted(set([z for pair in matrice_trafic.keys() for z in pair]))
    df_matrice = pd.DataFrame(0, index=zones, columns=zones)
    for (entree, sortie), count in matrice_trafic.items():
        df_matrice.loc[entree, sortie] = count
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_matrice, annot=True, fmt="d", cmap="YlGnBu", cbar=True)
    plt.title("Matrice Entrée → Sortie (Nombre de véhicules)")
    plt.xlabel("Sortie")
    plt.ylabel("Entrée")
    plt.tight_layout()
    plt.savefig("matrice_entree_sortie.png")
    plt.show()
    print("✔️ Matrice de trafic enregistrée dans matrice_entree_sortie.png")

# Programme principal
def main():
    video_path = "video.mp4"
    analyser_video(video_path)

    nb_vehicules = len(ids_vehicules)
    nb_personnes = len(ids_personnes)

    matrice_trafic = {}
    for traj in trajectoires.values():
        entree = traj["entree"]
        sortie = traj["sortie"]
        if entree and sortie:
            cle = (entree, sortie)
            matrice_trafic[cle] = matrice_trafic.get(cle, 0) + 1

    enregistrer_excel(nb_vehicules, nb_personnes, matrice_trafic)
    enregistrer_sqlite(nb_vehicules, nb_personnes)
    afficher_graphique(nb_vehicules, nb_personnes)
    afficher_matrice_trafic(matrice_trafic)

    print(f"\n🧾 Résumé :\n- Véhicules détectés : {nb_vehicules}\n- Personnes détectées : {nb_personnes}")
    print("\nMatrice Entrée → Sortie :")
    for (entree, sortie), count in matrice_trafic.items():
        print(f"{entree} → {sortie} : {count} véhicules")

if __name__ == "__main__":
    main()
