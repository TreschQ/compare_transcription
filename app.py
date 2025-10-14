import streamlit as st
import os
import glob
import difflib
import re
from pathlib import Path
import pandas as pd

st.set_page_config(
    page_title="Comparateur de Transcriptions",
    page_icon="📝",
    layout="wide"
)

def load_transcript_files(base_directory):
    """Charge tous les fichiers de transcription depuis les dossiers"""
    # Chercher dans output/videos/ pour les originaux
    videos_dir = os.path.join(base_directory, 'videos')
    improved_dir = os.path.join(base_directory, 'improved_transcripts')
    
    st.write(f"DEBUG: Cherche dans {videos_dir} et {improved_dir}")
    
    transcript_files = glob.glob(os.path.join(videos_dir, 'transcript_formatted_*.txt'))
    improved_files = glob.glob(os.path.join(improved_dir, 'improved_*.txt'))
    
    st.write(f"DEBUG: Trouvé {len(transcript_files)} fichiers originaux")
    st.write(f"DEBUG: Trouvé {len(improved_files)} fichiers améliorés")
    
    files_data = {}
    
    # Charger les fichiers originaux
    for file_path in transcript_files:
        filename = os.path.basename(file_path)
        base_name = filename.replace('transcript_formatted_', '').replace('.txt', '')
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        files_data[base_name] = {
            'original': content,
            'original_path': file_path,
            'improved': None,
            'improved_path': None
        }
    
    # Charger les fichiers améliorés
    for file_path in improved_files:
        filename = os.path.basename(file_path)
        base_name = filename.replace('improved_', '').replace('.txt', '')
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if base_name in files_data:
            files_data[base_name]['improved'] = content
            files_data[base_name]['improved_path'] = file_path
    
    st.write(f"DEBUG: {len(files_data)} fichiers appariés")
    
    return files_data

def find_common_start(text1, text2, min_words=4):
    """Trouve le point de départ commun entre deux textes (minimum 4-5 mots)"""
    words1 = text1.split()
    words2 = text2.split()
    
    # Chercher une séquence de mots communs
    for i in range(len(words1) - min_words + 1):
        for j in range(len(words2) - min_words + 1):
            # Vérifier si on a au moins min_words mots identiques consécutifs
            match_count = 0
            for k in range(min(len(words1) - i, len(words2) - j)):
                if words1[i + k].lower() == words2[j + k].lower():
                    match_count += 1
                else:
                    break
            
            if match_count >= min_words:
                # Retourner les indices de début dans les textes originaux
                start1 = len(' '.join(words1[:i]))
                start2 = len(' '.join(words2[:j]))
                if start1 > 0:
                    start1 += 1  # Ajouter l'espace
                if start2 > 0:
                    start2 += 1  # Ajouter l'espace
                return start1, start2
    
    return 0, 0

def tokenize_text(text):
    """Tokenise le texte en mots et ponctuation"""
    # Séparer les mots et la ponctuation
    tokens = re.findall(r'\w+|[^\w\s]', text)
    return tokens

def align_and_compare_texts(original, improved, min_common_words=4):
    """Aligne deux textes et génère une comparaison avec surbrillance"""
    
    # Trouver le point de départ commun
    start1, start2 = find_common_start(original, improved, min_common_words)
    
    # Extraire les parties alignées
    aligned_original = original[start1:]
    aligned_improved = improved[start2:]
    
    # Tokeniser
    tokens_original = tokenize_text(aligned_original)
    tokens_improved = tokenize_text(aligned_improved)
    
    # Utiliser difflib pour comparer
    matcher = difflib.SequenceMatcher(None, tokens_original, tokens_improved)
    
    original_html = []
    improved_html = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Tokens identiques
            for i in range(i1, i2):
                original_html.append(f'<span style="color: #cccccc;">{tokens_original[i]}</span>')
            for j in range(j1, j2):
                improved_html.append(f'<span style="color: #cccccc;">{tokens_improved[j]}</span>')
        
        elif tag == 'replace':
            # Tokens modifiés
            for i in range(i1, i2):
                original_html.append(f'<span style="background-color: #8B0000; color: #ffffff; padding: 2px 4px; border-radius: 3px; font-weight: bold;">{tokens_original[i]}</span>')
            for j in range(j1, j2):
                improved_html.append(f'<span style="background-color: #1E3A8A; color: #ffffff; padding: 2px 4px; border-radius: 3px; font-weight: bold;">{tokens_improved[j]}</span>')
        
        elif tag == 'delete':
            # Tokens supprimés
            for i in range(i1, i2):
                original_html.append(f'<span style="background-color: #8B0000; color: #ffffff; text-decoration: line-through; padding: 2px 4px; border-radius: 3px;">{tokens_original[i]}</span>')
        
        elif tag == 'insert':
            # Tokens ajoutés
            for j in range(j1, j2):
                improved_html.append(f'<span style="background-color: #166534; color: #ffffff; padding: 2px 4px; border-radius: 3px; font-weight: bold;">{tokens_improved[j]}</span>')
    
    # Rejoindre les tokens avec des espaces appropriés
    original_result = rebuild_text_with_spacing(original_html)
    improved_result = rebuild_text_with_spacing(improved_html)
    
    return original_result, improved_result, start1, start2

def rebuild_text_with_spacing(token_list):
    """Reconstruit le texte avec un espacement approprié"""
    if not token_list:
        return ""
    
    result = []
    for i, token in enumerate(token_list):
        # Extraire le contenu du token (enlever les balises HTML pour l'analyse)
        content = re.sub(r'<[^>]+>', '', token)
        
        if i == 0:
            result.append(token)
        else:
            # Ajouter un espace avant sauf pour la ponctuation
            if content not in '.,!?;:)]}' and not content.startswith("'"):
                result.append(' ')
            result.append(token)
    
    return ''.join(result)

def calculate_diff_stats(original, improved):
    """Calcule les statistiques de différences"""
    tokens_original = tokenize_text(original)
    tokens_improved = tokenize_text(improved)
    
    matcher = difflib.SequenceMatcher(None, tokens_original, tokens_improved)
    
    equal_count = 0
    replace_count = 0
    delete_count = 0
    insert_count = 0
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            equal_count += i2 - i1
        elif tag == 'replace':
            replace_count += max(i2 - i1, j2 - j1)
        elif tag == 'delete':
            delete_count += i2 - i1
        elif tag == 'insert':
            insert_count += j2 - j1
    
    total_tokens = len(tokens_original)
    changed_tokens = replace_count + delete_count + insert_count
    
    return {
        'total_original': len(tokens_original),
        'total_improved': len(tokens_improved),
        'equal': equal_count,
        'replaced': replace_count,
        'deleted': delete_count,
        'inserted': insert_count,
        'change_percentage': (changed_tokens / total_tokens * 100) if total_tokens > 0 else 0
    }

# Interface Streamlit
st.title("📝 Comparateur de Transcriptions")
st.markdown("Comparez les transcriptions originales avec les versions améliorées par l'IA")

# Sélection du dossier
col1, col2 = st.columns([2, 1])

with col1:
    directory = st.text_input(
        "Dossier contenant les transcriptions:",
        value="output/",
        help="Dossier contenant les fichiers transcript_formatted_*.txt et improved_*.txt"
    )

with col2:
    min_common_words = st.slider(
        "Mots communs minimum:",
        min_value=3,
        max_value=8,
        value=4,
        help="Nombre minimum de mots consécutifs identiques pour l'alignement"
    )

# Charger les fichiers si le dossier existe
if os.path.exists(directory):
    files_data = load_transcript_files(directory)
    
    if files_data:
        # Sélection du fichier
        available_files = [name for name, data in files_data.items() if data['improved'] is not None]
        
        if available_files:
            selected_file = st.selectbox(
                "Sélectionnez un fichier à comparer:",
                available_files,
                help="Seuls les fichiers ayant une version améliorée sont affichés"
            )
            
            if selected_file:
                data = files_data[selected_file]
                original_text = data['original']
                improved_text = data['improved']
                
                # Afficher les informations du fichier
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Longueur originale", f"{len(original_text):,} caractères")
                
                with col2:
                    st.metric("Longueur améliorée", f"{len(improved_text):,} caractères")
                
                with col3:
                    diff = len(improved_text) - len(original_text)
                    st.metric("Différence", f"{diff:+,} caractères")
                
                # Effectuer la comparaison
                if st.button("🔍 Comparer les textes", type="primary"):
                    with st.spinner("Analyse en cours..."):
                        original_html, improved_html, start1, start2 = align_and_compare_texts(
                            original_text, improved_text, min_common_words
                        )
                        
                        # Calculer les statistiques
                        stats = calculate_diff_stats(original_text[start1:], improved_text[start2:])
                        
                        # Afficher les statistiques
                        st.markdown("### 📊 Statistiques de comparaison")
                        
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        with col1:
                            st.metric("Tokens identiques", stats['equal'])
                        
                        with col2:
                            st.metric("Tokens modifiés", stats['replaced'])
                        
                        with col3:
                            st.metric("Tokens supprimés", stats['deleted'])
                        
                        with col4:
                            st.metric("Tokens ajoutés", stats['inserted'])
                        
                        with col5:
                            st.metric("% de changement", f"{stats['change_percentage']:.1f}%")
                        
                        if start1 > 0 or start2 > 0:
                            st.info(f"Alignement automatique détecté. Début de comparaison : position {start1} (original) / {start2} (amélioré)")
                        
                        # Afficher la comparaison côte à côte avec scroll synchronisé
                        st.markdown("### 🔄 Comparaison détaillée")
                        
                        # Composant HTML complet avec JavaScript intégré
                        sync_compare_html = f"""
                        <div style="display: flex; gap: 20px; margin: 20px 0;">
                            <div style="flex: 1;">
                                <h4 style="color: #ffffff; margin-bottom: 10px;">📄 Transcription originale</h4>
                                <div id="original-transcript" style="border: 2px solid #444; padding: 15px; border-radius: 8px; height: 400px; overflow-y: auto; background-color: #1e1e1e; color: #ffffff; font-family: monospace; line-height: 1.6;">
                                    {original_html}
                                </div>
                            </div>
                            <div style="flex: 1;">
                                <h4 style="color: #ffffff; margin-bottom: 10px;">✨ Transcription améliorée</h4>
                                <div id="improved-transcript" style="border: 2px solid #444; padding: 15px; border-radius: 8px; height: 400px; overflow-y: auto; background-color: #1e1e1e; color: #ffffff; font-family: monospace; line-height: 1.6;">
                                    {improved_html}
                                </div>
                            </div>
                        </div>
                        
                        <script>
                        (function() {{
                            let syncInProgress = false;
                            
                            function setupSyncScroll() {{
                                const original = document.getElementById('original-transcript');
                                const improved = document.getElementById('improved-transcript');
                                
                                if (original && improved) {{
                                    original.addEventListener('scroll', function() {{
                                        if (!syncInProgress) {{
                                            syncInProgress = true;
                                            improved.scrollTop = this.scrollTop;
                                            setTimeout(() => {{ syncInProgress = false; }}, 10);
                                        }}
                                    }});
                                    
                                    improved.addEventListener('scroll', function() {{
                                        if (!syncInProgress) {{
                                            syncInProgress = true;
                                            original.scrollTop = this.scrollTop;
                                            setTimeout(() => {{ syncInProgress = false; }}, 10);
                                        }}
                                    }});
                                    
                                    console.log('Scroll synchronization enabled');
                                }} else {{
                                    console.log('Elements not found, retrying...');
                                    setTimeout(setupSyncScroll, 200);
                                }}
                            }}
                            
                            // Démarrer après chargement complet
                            if (document.readyState === 'loading') {{
                                document.addEventListener('DOMContentLoaded', setupSyncScroll);
                            }} else {{
                                setupSyncScroll();
                            }}
                        }})();
                        </script>
                        """
                        
                        st.markdown(sync_compare_html, unsafe_allow_html=True)
                        
                        # Légende
                        st.markdown("### 🎨 Légende des couleurs")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.markdown('<span style="color: #cccccc; padding: 2px 5px; border: 1px solid #666; border-radius: 3px;">Identique</span>', unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown('<span style="background-color: #8B0000; color: #ffffff; padding: 2px 5px; border-radius: 3px;">Modifié (original)</span>', unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown('<span style="background-color: #1E3A8A; color: #ffffff; padding: 2px 5px; border-radius: 3px;">Modifié (amélioré)</span>', unsafe_allow_html=True)
                        
                        with col4:
                            st.markdown('<span style="background-color: #166534; color: #ffffff; padding: 2px 5px; border-radius: 3px;">Ajouté</span>', unsafe_allow_html=True)
        
        else:
            st.warning("Aucun fichier avec version améliorée trouvé dans ce dossier.")
    
    else:
        st.warning("Aucun fichier de transcription trouvé dans ce dossier.")

else:
    st.error(f"Le dossier '{directory}' n'existe pas.")

# Sidebar avec informations
st.sidebar.markdown("## ℹ️ Information")
st.sidebar.markdown("""
Cette application compare les transcriptions originales avec leurs versions améliorées par l'IA.

**Fonctionnalités :**
- 🔍 Alignement automatique des textes
- 🎨 Surbrillance des différences
- 📊 Statistiques de comparaison
- 🔄 Comparaison côte à côte

**Couleurs :**
- 🟢 Vert : Texte identique
- 🔴 Rouge : Texte modifié/supprimé
- 🔵 Bleu : Texte modifié (version améliorée)
- 🟡 Vert clair : Texte ajouté
""")