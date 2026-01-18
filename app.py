import cv2
import streamlit as st
from ultralytics import YOLO
from collections import defaultdict
import tempfile
import os

# Configuration de la page Streamlit
st.set_page_config(page_title="YOLO Animal Tracker", layout="wide")

st.title("🐾 YOLO Animal Tracking and Counting System")
st.markdown("---")

# Sidebar pour les paramètres
with st.sidebar:
    st.header("⚙️ Paramètres")
    
    # Upload de vidéo
    uploaded_file = st.file_uploader("Choisir une vidéo", type=['mp4', 'avi', 'mov'])
    
    # Sélection des classes d'animaux
    st.subheader("Classes d'animaux")
    animal_options = {
        'person': 0,
        'bird': 14,
        'cat': 15,
        'dog': 16,
        'horse': 17,
        'sheep': 18,
        'cow': 19
    }
    
    selected_animals = []
    for animal_name, animal_id in animal_options.items():
        if st.checkbox(animal_name, value=True if animal_id == 0 else False):
            selected_animals.append(animal_id)
    
    # Position de la ligne rouge
    st.subheader("Paramètres de la ligne")
    line_x_red = st.slider("Position X de la ligne rouge", 100, 800, 450)
    
    # Modèle YOLO
    st.subheader("Modèle YOLO")
    model_option = st.selectbox(
        "Choisir le modèle",
        ['yolo11l.pt', 'yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11x.pt']
    )

# Initialiser les variables de session
if 'class_counts' not in st.session_state:
    st.session_state.class_counts = defaultdict(int)
if 'crossed_ids' not in st.session_state:
    st.session_state.crossed_ids = set()
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Zones d'affichage principales
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Vue de la vidéo")
    video_placeholder = st.empty()
    
with col2:
    st.subheader("📊 Statistiques")
    stats_placeholder = st.empty()

# Fonction de traitement vidéo
def process_video(video_path, animal_classes, line_x_pos):
    # Charger le modèle YOLO
    model = YOLO(model_option)
    class_list = model.names
    
    cap = cv2.VideoCapture(video_path)
    
    # Réinitialiser les compteurs
    st.session_state.class_counts = defaultdict(int)
    st.session_state.crossed_ids = set()
    
    # Obtenir les propriétés de la vidéo pour l'affichage
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    while cap.isOpened() and st.session_state.processing:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Exécuter le tracking YOLO
        results = model.track(frame, persist=True, classes=animal_classes)
        
        if results[0].boxes.data is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
            class_indices = results[0].boxes.cls.int().tolist()
            confidences = results[0].boxes.conf.cpu()
            
            # Dessiner la ligne rouge
            cv2.line(frame, (line_x_pos, 0), (line_x_pos, frame_height), (0, 0, 255), 3)
            cv2.putText(frame, 'Red Line', (line_x_pos, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Traiter chaque détection
            for i, (box, class_id, conf) in enumerate(zip(boxes, class_indices, confidences)):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                class_name = class_list[class_id]
                
                # Afficher les informations de tracking
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                
                # Gérer les IDs de tracking
                track_id = track_ids[i] if i < len(track_ids) else i
                cv2.putText(frame, f"ID: {track_id} {class_name}", 
                          (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Vérifier si l'objet a traversé la ligne rouge
                if cx < line_x_pos and track_id not in st.session_state.crossed_ids:
                    st.session_state.crossed_ids.add(track_id)
                    st.session_state.class_counts[class_name] += 1
        
        # Convertir BGR to RGB pour Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Afficher la vidéo
        video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
        
        # Mettre à jour les statistiques
        stats_text = "### Comptage par classe:\n"
        for class_name, count in st.session_state.class_counts.items():
            stats_text += f"**{class_name}**: {count}\n"
        
        stats_text += f"\n**Total**: {sum(st.session_state.class_counts.values())}"
        stats_placeholder.markdown(stats_text)
    
    cap.release()

# Contrôles de traitement
col_controls = st.columns([1, 1, 1, 1])

with col_controls[0]:
    if uploaded_file and st.button("▶️ Démarrer le traitement"):
        # Sauvegarder la vidéo temporairement
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        st.session_state.processing = True
        process_video(video_path, selected_animals, line_x_red)
        os.unlink(video_path)  # Nettoyer le fichier temporaire

with col_controls[1]:
    if st.button("⏹️ Arrêter"):
        st.session_state.processing = False

with col_controls[2]:
    if st.button("🔄 Réinitialiser les compteurs"):
        st.session_state.class_counts = defaultdict(int)
        st.session_state.crossed_ids = set()
        stats_placeholder.empty()

# Informations supplémentaires
with st.expander("ℹ️ À propos de l'application"):
    st.markdown("""
    ### Fonctionnalités :
    1. **Tracking d'animaux** : Détection et suivi des animaux en temps réel
    2. **Comptage intelligent** : Compte les animaux qui traversent la ligne rouge
    3. **Interface intuitive** : Contrôles simples et visualisation claire
    
    ### Instructions :
    1. Téléchargez une vidéo dans la barre latérale
    2. Sélectionnez les classes d'animaux à détecter
    3. Ajustez la position de la ligne de comptage
    4. Cliquez sur "Démarrer le traitement"
    
    ### Classes disponibles :
    - Personne (0)
    - Oiseau (14)
    - Chat (15)
    - Chien (16)
    - Cheval (17)
    - Mouton (18)
    - Vache (19)
    """)

# Style CSS personnalisé
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        margin: 5px 0;
    }
    .stSlider > div {
        padding: 10px 0;
    }
</style>
""", unsafe_allow_html=True)