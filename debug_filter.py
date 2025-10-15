#!/usr/bin/env python3
"""Script de debug pour vérifier le filtrage dans l'app"""

import sys
import os
sys.path.append('.')

from app import filter_transcription_segments, load_transcript_files

def main():
    print("=== DEBUG FILTRAGE DANS L'APP ===")
    
    # Charger les fichiers comme dans l'app
    directory = "output/"
    files_data = load_transcript_files(directory)
    
    # Tester sur le fichier spécifique
    file_key = "manifest [manifest] copie"
    
    if file_key in files_data:
        data = files_data[file_key]
        original_text = data['original']
        improved_text = data['improved']
        
        print(f"Fichier trouvé: {file_key}")
        print(f"Texte original: {len(original_text)} caractères")
        print(f"Texte amélioré: {len(improved_text)} caractères")
        
        # Appliquer le filtrage
        print("\n--- Application du filtrage ---")
        filtered_text = filter_transcription_segments(original_text, improved_text)
        
        print(f"Texte filtré: {len(filtered_text)} caractères")
        print(f"Réduction: {len(improved_text) - len(filtered_text)} caractères")
        
        print(f"\nDébut du texte original: {original_text[:100]}...")
        print(f"\nDébut du texte amélioré: {improved_text[:100]}...")
        print(f"\nDébut du texte filtré: {filtered_text[:100]}...")
        
        # Vérifier si le filtrage a bien fonctionné
        if len(filtered_text) < len(improved_text):
            print("\n✅ Filtrage réussi!")
        else:
            print("\n❌ Aucun filtrage appliqué")
            
    else:
        print(f"Fichier '{file_key}' non trouvé dans les données")
        print("Fichiers disponibles:")
        for key in files_data.keys():
            print(f"  - {key}")

if __name__ == "__main__":
    main()