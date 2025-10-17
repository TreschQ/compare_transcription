import streamlit as st
import os
import glob
import difflib
import re
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Comparateur de Transcriptions",
    page_icon="📝",
    layout="wide"
)

def get_available_scopes(base_directory):
    """Récupère la liste des périmètres disponibles"""
    transcripts_dir = os.path.join(base_directory, 'transcripts')
    improved_dir = os.path.join(base_directory, 'improved_transcripts')
    
    scopes = set()
    
    # Chercher les dossiers dans transcripts et improved_transcripts
    if os.path.exists(transcripts_dir):
        scopes.update([d for d in os.listdir(transcripts_dir) if os.path.isdir(os.path.join(transcripts_dir, d))])
    
    if os.path.exists(improved_dir):
        scopes.update([d for d in os.listdir(improved_dir) if os.path.isdir(os.path.join(improved_dir, d))])
    
    return sorted(list(scopes))

def load_transcript_files(base_directory, selected_scope=None):
    """Charge tous les fichiers de transcription depuis les dossiers par périmètre"""
    transcripts_dir = os.path.join(base_directory, 'transcripts')
    improved_dir = os.path.join(base_directory, 'improved_transcripts')
    
    files_data = {}
    
    if selected_scope == "TOUS":
        # Charger tous les périmètres
        scopes = get_available_scopes(base_directory)
        scopes = [s for s in scopes if s != "TOUS"]
    elif selected_scope:
        # Charger un périmètre spécifique
        scopes = [selected_scope]
    else:
        # Mode rétrocompatible (recherche dans le dossier racine)
        videos_dir = os.path.join(base_directory, 'videos')
        
        transcript_files = glob.glob(os.path.join(videos_dir, 'transcript_formatted_*.txt'))
        improved_files = glob.glob(os.path.join(improved_dir, 'improved_*.txt'))
        
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
                'improved_path': None,
                'scope': 'LEGACY'
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
        
        return files_data
    
    # Charger les fichiers par périmètre
    for scope in scopes:
        scope_transcripts_dir = os.path.join(transcripts_dir, scope)
        scope_improved_dir = os.path.join(improved_dir, scope)
        
        if os.path.exists(scope_transcripts_dir):
            transcript_files = glob.glob(os.path.join(scope_transcripts_dir, 'transcript_formatted_*.txt'))
        else:
            transcript_files = []
            
        if os.path.exists(scope_improved_dir):
            improved_files = glob.glob(os.path.join(scope_improved_dir, 'improved_*.txt'))
        else:
            improved_files = []
        
        # Charger les fichiers originaux du périmètre
        for file_path in transcript_files:
            filename = os.path.basename(file_path)
            base_name = filename.replace('transcript_formatted_', '').replace('.txt', '')
            file_key = f"{scope}:{base_name}" if selected_scope == "TOUS" else base_name
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            files_data[file_key] = {
                'original': content,
                'original_path': file_path,
                'improved': None,
                'improved_path': None,
                'scope': scope
            }
        
        # Charger les fichiers améliorés du périmètre
        for file_path in improved_files:
            filename = os.path.basename(file_path)
            base_name = filename.replace('improved_', '').replace('.txt', '')
            file_key = f"{scope}:{base_name}" if selected_scope == "TOUS" else base_name
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if file_key in files_data:
                files_data[file_key]['improved'] = content
                files_data[file_key]['improved_path'] = file_path
    
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

def filter_transcription_segments(original_text, improved_text):
    """
    Filtre la transcription améliorée selon des règles strictes :
    - Début : Commencer seulement si 90% des 30 premiers tokens correspondent
    - Fin : Arrêter dès qu'on trouve 30 tokens consécutifs non correspondants
    
    Args:
        original_text (str): Texte original
        improved_text (str): Texte amélioré 
    
    Returns:
        str: Texte amélioré filtré
    """
    if not original_text or not improved_text:
        return improved_text
    
    # Tokeniser comme dans l'app
    tokens_original = tokenize_text(original_text)
    tokens_improved = tokenize_text(improved_text)
    
    if len(tokens_original) < 30 or len(tokens_improved) < 30:
        return improved_text
    
    # 1. Trouver le point de départ (90% de correspondance sur 30 tokens)
    start_pos = None
    original_start_30 = tokens_original[:30]
    
    for i in range(len(tokens_improved) - 29):
        improved_segment_30 = tokens_improved[i:i+30]
        
        # Calculer la correspondance exacte
        matches = sum(1 for orig, imp in zip(original_start_30, improved_segment_30) if orig.lower() == imp.lower())
        similarity = matches / 30
        
        if similarity >= 0.9:  # 90% de correspondance
            start_pos = i
            break
    
    if start_pos is None:
        return improved_text
    
    # 2. Utiliser difflib comme dans l'app pour détecter les changements
    matcher = difflib.SequenceMatcher(None, tokens_original, tokens_improved[start_pos:])
    
    end_pos = len(tokens_improved)
    consecutive_non_equal = 0
    current_pos = start_pos
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            consecutive_non_equal = 0
            current_pos = start_pos + j2
        else:
            # Compter les tokens non correspondants
            non_equal_count = max(i2 - i1, j2 - j1)
            consecutive_non_equal += non_equal_count
            
            if consecutive_non_equal >= 30:
                end_pos = start_pos + j1  # Position où ont commencé les non-correspondances
                break
            
            current_pos = start_pos + j2
    
    # 3. Extraire le texte filtré en préservant le formatage
    if start_pos > 0 or end_pos < len(tokens_improved):
        # Reconstruire le texte avec le bon espacement
        filtered_tokens = tokens_improved[start_pos:end_pos]
        return rebuild_text_with_spacing(filtered_tokens)
    
    return improved_text

def calculate_all_diff_percentages(files_data):
    """Calcule les pourcentages d'écart pour toutes les transcriptions (avec filtrage automatique)"""
    percentages = []
    file_names = []
    scopes = []
    all_stats = []
    
    for file_name, data in files_data.items():
        if data['improved'] is not None:
            # Appliquer le filtrage automatiquement
            filtered_improved = filter_transcription_segments(data['original'], data['improved'])
            
            # Aligner les textes
            start1, start2 = find_common_start(data['original'], filtered_improved, 4)
            aligned_original = data['original'][start1:]
            aligned_improved = filtered_improved[start2:]
            
            # Calculer les stats
            stats = calculate_diff_stats(aligned_original, aligned_improved)
            percentages.append(stats['change_percentage'])
            file_names.append(file_name)
            scopes.append(data.get('scope', 'Unknown'))
            all_stats.append(stats)
    
    return percentages, file_names, scopes, all_stats

def calculate_global_token_stats(all_stats, scopes=None):
    """Calcule les statistiques globales de tokens"""
    if not all_stats:
        return None
    
    # Statistiques globales
    global_stats = {
        'total_files': len(all_stats),
        'total_tokens_original': sum(stat['total_original'] for stat in all_stats),
        'total_tokens_improved': sum(stat['total_improved'] for stat in all_stats),
        'total_equal': sum(stat['equal'] for stat in all_stats),
        'total_replaced': sum(stat['replaced'] for stat in all_stats),
        'total_deleted': sum(stat['deleted'] for stat in all_stats),
        'total_inserted': sum(stat['inserted'] for stat in all_stats)
    }
    
    # Calculer les pourcentages globaux
    total_changes = global_stats['total_replaced'] + global_stats['total_deleted'] + global_stats['total_inserted']
    global_stats['global_change_percentage'] = (total_changes / global_stats['total_tokens_original'] * 100) if global_stats['total_tokens_original'] > 0 else 0
    
    # Statistiques par périmètre si disponible
    scope_stats = {}
    if scopes and len(set(scopes)) > 1:
        unique_scopes = list(set(scopes))
        for scope in unique_scopes:
            scope_indices = [i for i, s in enumerate(scopes) if s == scope]
            scope_data = [all_stats[i] for i in scope_indices]
            
            scope_stats[scope] = {
                'files': len(scope_data),
                'total_original': sum(stat['total_original'] for stat in scope_data),
                'total_improved': sum(stat['total_improved'] for stat in scope_data),
                'equal': sum(stat['equal'] for stat in scope_data),
                'replaced': sum(stat['replaced'] for stat in scope_data),
                'deleted': sum(stat['deleted'] for stat in scope_data),
                'inserted': sum(stat['inserted'] for stat in scope_data)
            }
            
            total_scope_changes = scope_stats[scope]['replaced'] + scope_stats[scope]['deleted'] + scope_stats[scope]['inserted']
            scope_stats[scope]['change_percentage'] = (total_scope_changes / scope_stats[scope]['total_original'] * 100) if scope_stats[scope]['total_original'] > 0 else 0
    
    return global_stats, scope_stats

def create_token_statistics_charts(all_stats, scopes=None):
    """Crée des graphiques pour les statistiques de tokens"""
    if not all_stats:
        return None, None, None
    
    # Préparer les données pour les graphiques
    data_for_charts = []
    for i, stat in enumerate(all_stats):
        scope = scopes[i] if scopes else 'All'
        data_for_charts.append({
            'Fichier': f"File {i+1}",
            'Périmètre': scope,
            'Supprimés': stat['deleted'],
            'Modifiés': stat['replaced'],
            'Ajoutés': stat['inserted'],
            'Identiques': stat['equal']
        })
    
    df_tokens = pd.DataFrame(data_for_charts)
    
    # Graphique en barres empilées pour les types de changements
    fig_stacked = go.Figure()
    
    colors = {
        'Supprimés': '#8B0000',
        'Modifiés': '#1E3A8A', 
        'Ajoutés': '#166534',
        'Identiques': '#cccccc'
    }
    
    for change_type in ['Supprimés', 'Modifiés', 'Ajoutés', 'Identiques']:
        fig_stacked.add_trace(go.Bar(
            name=change_type,
            x=df_tokens['Fichier'],
            y=df_tokens[change_type],
            marker_color=colors[change_type]
        ))
    
    fig_stacked.update_layout(
        title="Distribution des types de changements de tokens par fichier",
        xaxis_title="Fichiers",
        yaxis_title="Nombre de tokens",
        barmode='stack',
        showlegend=True
    )
    
    # Graphique en secteurs pour la répartition globale
    global_stats, _ = calculate_global_token_stats(all_stats, scopes)
    
    fig_pie = go.Figure(data=[go.Pie(
        labels=['Identiques', 'Modifiés', 'Supprimés', 'Ajoutés'],
        values=[global_stats['total_equal'], global_stats['total_replaced'], 
                global_stats['total_deleted'], global_stats['total_inserted']],
        hole=.3,
        marker_colors=[colors['Identiques'], colors['Modifiés'], colors['Supprimés'], colors['Ajoutés']]
    )])
    
    fig_pie.update_layout(
        title="Répartition globale des types de changements",
        annotations=[dict(text='Tokens', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )
    
    # Graphique de distribution des changements par périmètre (si applicable)
    fig_scope = None
    if scopes and len(set(scopes)) > 1:
        scope_totals = df_tokens.groupby('Périmètre')[['Supprimés', 'Modifiés', 'Ajoutés']].sum()
        
        fig_scope = go.Figure()
        for change_type in ['Supprimés', 'Modifiés', 'Ajoutés']:
            fig_scope.add_trace(go.Bar(
                name=change_type,
                x=scope_totals.index,
                y=scope_totals[change_type],
                marker_color=colors[change_type]
            ))
        
        fig_scope.update_layout(
            title="Total des changements par périmètre",
            xaxis_title="Périmètres",
            yaxis_title="Nombre de tokens",
            barmode='group'
        )
    
    return fig_stacked, fig_pie, fig_scope

def create_distribution_chart(percentages, file_names, scopes=None):
    """Crée un graphique de distribution des écarts de tokens"""
    df_data = {
        'Fichier': file_names,
        'Écart_pourcentage': percentages
    }
    
    if scopes:
        df_data['Périmètre'] = scopes
    
    df = pd.DataFrame(df_data)
    
    # Histogramme
    if scopes and len(set(scopes)) > 1:
        fig_hist = px.histogram(
            df, 
            x='Écart_pourcentage',
            color='Périmètre',
            nbins=20,
            title="Distribution des écarts de tokens (%) par périmètre",
            labels={'Écart_pourcentage': 'Écart de tokens (%)', 'count': 'Nombre de fichiers'}
        )
    else:
        fig_hist = px.histogram(
            df, 
            x='Écart_pourcentage',
            nbins=20,
            title="Distribution des écarts de tokens (%)",
            labels={'Écart_pourcentage': 'Écart de tokens (%)', 'count': 'Nombre de fichiers'},
            color_discrete_sequence=['#1f77b4']
        )
    
    fig_hist.update_layout(
        xaxis_title="Écart de tokens (%)",
        yaxis_title="Nombre de fichiers"
    )
    
    # Box plot
    if scopes and len(set(scopes)) > 1:
        fig_box = px.box(
            df,
            x='Périmètre',
            y='Écart_pourcentage',
            title="Répartition des écarts de tokens par périmètre",
            labels={'Écart_pourcentage': 'Écart de tokens (%)', 'Périmètre': 'Périmètre'}
        )
        fig_box.update_layout(
            xaxis_title="Périmètre",
            yaxis_title="Écart de tokens (%)"
        )
    else:
        fig_box = px.box(
            df,
            y='Écart_pourcentage',
            title="Répartition des écarts de tokens",
            labels={'Écart_pourcentage': 'Écart de tokens (%)'}
        )
        fig_box.update_layout(
            yaxis_title="Écart de tokens (%)",
            showlegend=False
        )
    
    # Graphique en barres détaillé
    if scopes and len(set(scopes)) > 1:
        fig_bar = px.bar(
            df.sort_values('Écart_pourcentage', ascending=False),
            x='Fichier',
            y='Écart_pourcentage',
            color='Périmètre',
            title="Écarts de tokens par fichier et périmètre",
            labels={'Écart_pourcentage': 'Écart de tokens (%)', 'Fichier': 'Nom du fichier'}
        )
    else:
        fig_bar = px.bar(
            df.sort_values('Écart_pourcentage', ascending=False),
            x='Fichier',
            y='Écart_pourcentage',
            title="Écarts de tokens par fichier",
            labels={'Écart_pourcentage': 'Écart de tokens (%)', 'Fichier': 'Nom du fichier'},
            color='Écart_pourcentage',
            color_continuous_scale='Viridis'
        )
    
    fig_bar.update_layout(
        xaxis_title="Fichiers",
        yaxis_title="Écart de tokens (%)",
        xaxis={'tickangle': 45}
    )
    
    return fig_hist, fig_box, fig_bar, df

def load_nlp_analysis(base_directory, scope):
    """Charge les analyses NLP pour un périmètre donné"""
    if scope == "TOUS" or not scope:
        return None
    
    analysis_file = os.path.join(base_directory, 'improved_transcripts', scope, f'nlp_analyses_{scope}.json')
    
    if not os.path.exists(analysis_file):
        return None
    
    try:
        with open(analysis_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Erreur lors du chargement de l'analyse NLP : {e}")
        return None

def calculate_nlp_kpis(nlp_data):
    """Calcule les KPI moyens des types de changements à partir des analyses NLP"""
    if not nlp_data or 'analyses' not in nlp_data:
        return None
    
    all_change_types = []
    
    for analysis in nlp_data['analyses']:
        if 'report_summary' in analysis and 'change_types' in analysis['report_summary']:
            change_types = analysis['report_summary']['change_types']
            all_change_types.append(change_types)
    
    if not all_change_types:
        return None
    
    # Calculer les moyennes pour chaque type de changement
    df = pd.DataFrame(all_change_types)
    means = df.mean()
    
    # Calculer le total moyen et les pourcentages
    total_mean = means.sum()
    percentages = (means / total_mean * 100) if total_mean > 0 else means * 0
    
    return {
        'means': means.to_dict(),
        'percentages': percentages.to_dict(),
        'total_files': len(all_change_types),
        'total_mean': total_mean
    }

def create_nlp_kpi_charts(kpi_data):
    """Crée des graphiques pour les KPI des types de changements NLP"""
    if not kpi_data:
        return None, None
    
    means = kpi_data['means']
    percentages = kpi_data['percentages']
    
    # Couleurs pour chaque type de changement
    colors = {
        'orthographic': '#2E8B57',  # Vert foncé
        'grammatical': '#4169E1',   # Bleu royal
        'punctuation': '#FF6347',   # Rouge tomate
        'lexical': '#DAA520',       # Or
        'structural': '#8A2BE2',    # Violet
        'additions': '#32CD32',     # Vert lime
        'deletions': '#DC143C'      # Rouge cramoisi
    }
    
    # Graphique en barres des moyennes
    fig_means = go.Figure()
    
    change_types = list(means.keys())
    values = list(means.values())
    bar_colors = [colors.get(ct, '#666666') for ct in change_types]
    
    fig_means.add_trace(go.Bar(
        x=change_types,
        y=values,
        marker_color=bar_colors,
        text=[f'{v:.1f}' for v in values],
        textposition='auto'
    ))
    
    fig_means.update_layout(
        title="Moyenne des types de changements par fichier (Analyse NLP)",
        xaxis_title="Types de changements",
        yaxis_title="Nombre moyen de changements",
        showlegend=False
    )
    
    # Graphique en secteurs des pourcentages
    fig_pie = go.Figure(data=[go.Pie(
        labels=change_types,
        values=list(percentages.values()),
        hole=.3,
        marker_colors=[colors.get(ct, '#666666') for ct in change_types],
        textinfo='label+percent',
        textposition='auto'
    )])
    
    fig_pie.update_layout(
        title="Répartition des types de changements (Analyse NLP)",
        annotations=[dict(text=f'Total: {kpi_data["total_mean"]:.1f}', x=0.5, y=0.5, font_size=16, showarrow=False)]
    )
    
    return fig_means, fig_pie

# Interface Streamlit
st.title("📝 Comparateur de Transcriptions")
st.markdown("Comparez les transcriptions originales avec les versions améliorées par l'IA")

# Sélection du dossier et périmètre
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    directory = st.text_input(
        "Dossier contenant les transcriptions:",
        value="output/",
        help="Dossier contenant les fichiers transcript_formatted_*.txt et improved_*.txt"
    )

with col2:
    # Sélection du périmètre
    if os.path.exists(directory):
        available_scopes = get_available_scopes(directory)
        if available_scopes:
            scope_options = ["TOUS"] + available_scopes
            selected_scope = st.selectbox(
                "Périmètre:",
                scope_options,
                help="Sélectionnez un périmètre spécifique ou tous les périmètres"
            )
        else:
            selected_scope = None
            st.warning("Aucun périmètre détecté")
    else:
        selected_scope = None

with col3:
    min_common_words = st.slider(
        "Mots communs minimum:",
        min_value=3,
        max_value=8,
        value=4,
        help="Nombre minimum de mots consécutifs identiques pour l'alignement"
    )

# Charger les fichiers si le dossier existe
if os.path.exists(directory):
    files_data = load_transcript_files(directory, selected_scope)
    
    if files_data:
        # Section Analyses des transcriptions
        
        available_files = [name for name, data in files_data.items() if data['improved'] is not None]
        
        if available_files:
            # Calculer les écarts pour tous les fichiers
            with st.spinner("Calcul des écarts pour tous les fichiers..."):
                percentages, file_names, scopes, all_stats = calculate_all_diff_percentages(files_data)
                
                if percentages:
                    # Afficher le périmètre sélectionné
                    if selected_scope:
                        if selected_scope == "TOUS":
                            st.info(f"📊 Analyse de tous les périmètres ({len(set(scopes))} périmètres: {', '.join(sorted(set(scopes)))})")
                        else:
                            st.info(f"📊 Analyse du périmètre: **{selected_scope}**")
                    
                    # === SECTION STATISTIQUES GLOBALES DE TOKENS ===
                    st.markdown("## 🔢 Statistiques globales des tokens")
                    
                    # Calculer les statistiques globales
                    global_stats, scope_stats = calculate_global_token_stats(all_stats, scopes)
                    
                    # Affichage des métriques principales
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("📁 Fichiers analysés", global_stats['total_files'])
                    
                    with col2:
                        st.metric("🔗 Tokens originaux", f"{global_stats['total_tokens_original']:,}")
                    
                    with col3:
                        st.metric("✨ Tokens améliorés", f"{global_stats['total_tokens_improved']:,}")
                    
                    with col4:
                        diff_tokens = global_stats['total_tokens_improved'] - global_stats['total_tokens_original']
                        st.metric("📈 Différence nette", f"{diff_tokens:+,}")
                    
                    with col5:
                        st.metric("🎯 % changement global", f"{global_stats['global_change_percentage']:.1f}%")
                    
                    # Métriques détaillées des changements
                    st.markdown("### 📊 Détail des changements de tokens")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("🟢 Identiques", f"{global_stats['total_equal']:,}")
                    
                    with col2:
                        st.metric("🔴 Supprimés", f"{global_stats['total_deleted']:,}")
                    
                    with col3:
                        st.metric("🔵 Modifiés", f"{global_stats['total_replaced']:,}")
                    
                    with col4:
                        st.metric("🟡 Ajoutés", f"{global_stats['total_inserted']:,}")
                    
                    # Graphiques des statistiques de tokens
                    st.markdown("### 📈 Visualisations des changements de tokens")
                    
                    fig_stacked, fig_pie, fig_scope = create_token_statistics_charts(all_stats, scopes)
                    
                    if fig_stacked and fig_pie:
                        tab_tokens1, tab_tokens2, tab_tokens3 = st.tabs(["🥧 Répartition globale", "📊 Par fichier", "🏛️ Par périmètre"])
                        
                        with tab_tokens1:
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        with tab_tokens2:
                            st.plotly_chart(fig_stacked, use_container_width=True)
                        
                        with tab_tokens3:
                            if fig_scope:
                                st.plotly_chart(fig_scope, use_container_width=True)
                            else:
                                st.info("Visualisation par périmètre disponible uniquement avec l'option 'TOUS'")
                    
                    # Tableau des statistiques par périmètre
                    if scope_stats and len(scope_stats) > 1:
                        st.markdown("### 🏛️ Statistiques détaillées par périmètre")
                        
                        scope_df_data = []
                        for scope, stats in scope_stats.items():
                            scope_df_data.append({
                                'Périmètre': scope,
                                'Fichiers': stats['files'],
                                'Tokens orig.': f"{stats['total_original']:,}",
                                'Tokens amél.': f"{stats['total_improved']:,}",
                                'Identiques': f"{stats['equal']:,}",
                                'Supprimés': f"{stats['deleted']:,}",
                                'Modifiés': f"{stats['replaced']:,}",
                                'Ajoutés': f"{stats['inserted']:,}",
                                '% changement': f"{stats['change_percentage']:.1f}%"
                            })
                        
                        scope_df = pd.DataFrame(scope_df_data)
                        st.dataframe(scope_df, use_container_width=True)
                    
                    # === SECTION KPI ANALYSES NLP ===
                    # Charger et afficher les KPI NLP si disponibles pour le périmètre sélectionné
                    if selected_scope and selected_scope != "TOUS":
                        nlp_data = load_nlp_analysis(directory, selected_scope)
                        if nlp_data:
                            st.markdown("## 🧠 KPI des Types de Changements (Analyse NLP)")
                            
                            kpi_data = calculate_nlp_kpis(nlp_data)
                            if kpi_data:
                                # Afficher les métriques principales
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("📄 Fichiers analysés (NLP)", kpi_data['total_files'])
                                
                                with col2:
                                    st.metric("📊 Changements moyens/fichier", f"{kpi_data['total_mean']:.1f}")
                                
                                with col3:
                                    top_type = max(kpi_data['means'], key=kpi_data['means'].get)
                                    st.metric("🏆 Type principal", top_type.title())
                                
                                with col4:
                                    top_percentage = kpi_data['percentages'][top_type]
                                    st.metric("📈 % du type principal", f"{top_percentage:.1f}%")
                                
                                # Tableaux détaillés des moyennes
                                st.markdown("### 📋 Détail des moyennes par type de changement")
                                
                                kpi_df_data = []
                                for change_type, mean_value in kpi_data['means'].items():
                                    percentage = kpi_data['percentages'][change_type]
                                    kpi_df_data.append({
                                        'Type de changement': change_type.title(),
                                        'Moyenne': f"{mean_value:.2f}",
                                        'Pourcentage': f"{percentage:.1f}%"
                                    })
                                
                                kpi_df = pd.DataFrame(kpi_df_data)
                                kpi_df = kpi_df.sort_values('Moyenne', key=lambda x: x.str.replace(',', '.').astype(float), ascending=False)
                                st.dataframe(kpi_df, use_container_width=True)
                                
                                # Graphiques des KPI NLP
                                fig_means, fig_pie = create_nlp_kpi_charts(kpi_data)
                                
                                if fig_means and fig_pie:
                                    tab_nlp1, tab_nlp2 = st.tabs(["📊 Moyennes par type", "🥧 Répartition"])
                                    
                                    with tab_nlp1:
                                        st.plotly_chart(fig_means, use_container_width=True)
                                    
                                    with tab_nlp2:
                                        st.plotly_chart(fig_pie, use_container_width=True)
                                
                                st.markdown("---")
                            else:
                                st.warning("⚠️ Impossible de calculer les KPI à partir des données NLP")
                        else:
                            st.info(f"ℹ️ Aucune analyse NLP disponible pour le périmètre **{selected_scope}**")
                    
                    # === SECTION DISTRIBUTION DES ÉCARTS ===
                    st.markdown("---")
                    st.markdown("## 📊 Distribution des pourcentages d'écarts")
                    
                    # Créer les graphiques
                    fig_hist, fig_box, fig_bar, df = create_distribution_chart(percentages, file_names, scopes)
                    
                    # Afficher les statistiques globales
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Nombre de fichiers", len(percentages))
                    
                    with col2:
                        st.metric("Écart moyen", f"{pd.Series(percentages).mean():.1f}%")
                    
                    with col3:
                        st.metric("Écart médian", f"{pd.Series(percentages).median():.1f}%")
                    
                    with col4:
                        st.metric("Écart max", f"{pd.Series(percentages).max():.1f}%")
                    
                    # Statistiques par périmètre si TOUS est sélectionné
                    if selected_scope == "TOUS" and len(set(scopes)) > 1:
                        st.markdown("### 📈 Statistiques par périmètre")
                        scope_stats = df.groupby('Périmètre')['Écart_pourcentage'].agg(['count', 'mean', 'median', 'max']).round(1)
                        scope_stats.columns = ['Nb fichiers', 'Moyenne (%)', 'Médiane (%)', 'Max (%)']
                        st.dataframe(scope_stats, use_container_width=True)
                    
                    # Onglets pour différentes visualisations
                    tab1, tab2, tab3, tab4 = st.tabs(["📈 Histogramme", "📦 Box Plot", "📊 Par fichier", "📋 Tableau"])
                    
                    with tab1:
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    with tab2:
                        st.plotly_chart(fig_box, use_container_width=True)
                    
                    with tab3:
                        st.plotly_chart(fig_bar, use_container_width=True)
                    
                    with tab4:
                        # Afficher le tableau des données
                        df_display = df.copy()
                        df_display['Écart_pourcentage'] = df_display['Écart_pourcentage'].round(2)
                        df_display = df_display.sort_values('Écart_pourcentage', ascending=False)
                        
                        if 'Périmètre' in df_display.columns:
                            df_display.columns = ['Fichier', 'Écart (%)', 'Périmètre']
                            df_display = df_display[['Périmètre', 'Fichier', 'Écart (%)']]
                        else:
                            df_display.columns = ['Fichier', 'Écart (%)']
                        
                        st.dataframe(df_display, use_container_width=True)
        
        # Section Comparaison détaillée
        st.markdown("---")
        st.markdown("## 🔍 Comparaison détaillée")
        
        if available_files:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Créer des options avec informations de périmètre pour l'affichage
                if selected_scope == "TOUS":
                    file_options = []
                    for file_name in available_files:
                        scope = files_data[file_name].get('scope', 'Unknown')
                        display_name = f"[{scope}] {file_name.split(':')[-1] if ':' in file_name else file_name}"
                        file_options.append(display_name)
                    
                    selected_display = st.selectbox(
                        "Sélectionnez un fichier à comparer:",
                        file_options,
                        help="Seuls les fichiers ayant une version améliorée sont affichés"
                    )
                    
                    # Retrouver le nom de fichier original
                    selected_index = file_options.index(selected_display)
                    selected_file = available_files[selected_index]
                else:
                    selected_file = st.selectbox(
                        "Sélectionnez un fichier à comparer:",
                        available_files,
                        help="Seuls les fichiers ayant une version améliorée sont affichés"
                    )
            
            with col2:
                disable_filter = st.checkbox(
                    "Désactiver le filtrage",
                    value=False,
                    help="Désactive le filtrage automatique pour voir le texte brut avec les métadonnées LLM"
                )
            
            if selected_file:
                data = files_data[selected_file]
                original_text = data['original']
                improved_text_raw = data['improved']
                
                # Appliquer le filtrage par défaut
                improved_text = filter_transcription_segments(original_text, improved_text_raw)
                reduction = len(improved_text_raw) - len(improved_text)
                
                if not disable_filter:
                    st.info(f"🔧 Filtrage activé par défaut - Réduction: {reduction:,} caractères ({reduction/len(improved_text_raw)*100:.1f}%)")
                else:
                    # Désactiver le filtrage si demandé
                    improved_text = improved_text_raw
                    st.info("⚠️ Filtrage désactivé - Affichage du texte brut avec métadonnées LLM")
                
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
- 🎯 **Sélection par périmètre** : BATCH1_MELTING, FINFO, OUTREMER, REGIONS
- 📊 **Distribution des écarts** : Analyse statistique globale et par périmètre
- 📈 **Visualisations interactives** : Histogrammes, box plots, graphiques
- 🔍 Alignement automatique des textes
- 🎨 Surbrillance des différences
- 📊 Statistiques de comparaison
- 🔄 Comparaison côte à côte

**Périmètres disponibles :**
- **TOUS** : Vue agrégée de tous les périmètres
- **BATCH1_MELTING** : Transcriptions du lot MELTING
- **FINFO** : Transcriptions FINFO
- **OUTREMER** : Transcriptions Outre-mer
- **REGIONS** : Transcriptions des régions

**Analyses disponibles :**
- 🔢 **Statistiques globales de tokens** : Nombre total de tokens supprimés, modifiés, ajoutés
- 📊 **Visualisations détaillées** : Graphiques en secteurs, barres empilées, distributions par périmètre
- 📈 **Métriques par périmètre** : Comparaison détaillée entre BATCH1_MELTING, FINFO, OUTREMER, REGIONS
- 📋 **Tableaux de synthèse** : Statistiques complètes par fichier et périmètre
- 🎯 **Pourcentages d'écarts** : Distribution, moyenne, médiane, maximum
- 🧠 **KPI Analyses NLP** : Types de changements moyens par périmètre (orthographique, grammatical, ponctuation, lexical, structurel, ajouts, suppressions)
""")