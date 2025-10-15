#!/usr/bin/env python3
"""Script de test pour la nouvelle fonction de filtrage des transcriptions"""

import re

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
    
    # Tokeniser les textes
    original_tokens = re.findall(r'\w+', original_text.lower())
    improved_tokens = re.findall(r'\w+', improved_text.lower())
    
    print(f"Tokens original: {len(original_tokens)}")
    print(f"Tokens amélioré: {len(improved_tokens)}")
    
    if len(original_tokens) < 30 or len(improved_tokens) < 30:
        print("Pas assez de tokens pour le filtrage")
        return improved_text
    
    # 1. Trouver le point de départ (90% de correspondance sur 30 tokens)
    start_pos = None
    original_start_30 = original_tokens[:30]
    
    print(f"\n30 premiers tokens originaux: {' '.join(original_start_30)}")
    print("\nRecherche du point de départ (90% correspondance sur 30 tokens):")
    
    for i in range(len(improved_tokens) - 29):
        improved_segment_30 = improved_tokens[i:i+30]
        
        # Calculer la correspondance
        matches = sum(1 for orig, imp in zip(original_start_30, improved_segment_30) if orig == imp)
        similarity = matches / 30
        
        if i < 5 or similarity >= 0.9:  # Afficher les 5 premiers + ceux qui passent le test
            print(f"Position {i}: {matches}/30 correspondances ({similarity:.1%}) - {' '.join(improved_segment_30[:5])}...")
        
        if similarity >= 0.9:  # 90% de correspondance
            start_pos = i
            print(f"✅ Point de départ trouvé à position {i} avec {similarity:.1%} correspondance")
            break
    
    if start_pos is None:
        print("❌ Aucun point de départ valide trouvé, retour du texte original")
        return improved_text
    
    # 2. Trouver le point de fin (30 tokens consécutifs non correspondants)
    end_pos = len(improved_tokens)
    
    print(f"\nRecherche du point de fin (arrêt à 30 tokens non correspondants consécutifs):")
    print(f"Début de comparaison depuis position {start_pos}")
    
    # Créer une fenêtre glissante pour comparer
    original_idx = 0
    improved_idx = start_pos
    consecutive_mismatches = 0
    max_consecutive = 0
    
    while improved_idx < len(improved_tokens) and original_idx < len(original_tokens):
        if improved_tokens[improved_idx] == original_tokens[original_idx]:
            if consecutive_mismatches > 0:
                print(f"Position {improved_idx}: Correspondance trouvée après {consecutive_mismatches} non-correspondances")
                max_consecutive = max(max_consecutive, consecutive_mismatches)
            consecutive_mismatches = 0
            original_idx += 1
        else:
            consecutive_mismatches += 1
            if consecutive_mismatches == 30:
                end_pos = improved_idx - 29  # Arrêter avant les 30 tokens non correspondants
                print(f"🛑 Arrêt détecté à position {improved_idx} (30 tokens non correspondants consécutifs)")
                break
        
        improved_idx += 1
    
    print(f"Max consecutive mismatches trouvés: {max_consecutive}")
    print(f"Position de fin: {end_pos} (sur {len(improved_tokens)} tokens)")
    
    # 3. Extraire le texte filtré en préservant le formatage original
    if start_pos > 0 or end_pos < len(improved_tokens):
        print(f"\nApplication du filtrage: tokens[{start_pos}:{end_pos}]")
        
        # Chercher les positions dans le texte original avec formatage
        words_pattern = ' '.join(improved_tokens[start_pos:start_pos+5])  # 5 premiers mots
        start_char_pos = improved_text.lower().find(words_pattern)
        
        if start_char_pos != -1 and end_pos < len(improved_tokens):
            # Chercher la fin
            end_words_pattern = ' '.join(improved_tokens[max(0, end_pos-5):end_pos])  # 5 derniers mots
            end_char_pos = improved_text.lower().find(end_words_pattern, start_char_pos)
            
            if end_char_pos != -1:
                end_char_pos += len(end_words_pattern)
                result = improved_text[start_char_pos:end_char_pos]
                print(f"✅ Filtrage réussi: {len(result)} caractères (réduction: {len(improved_text) - len(result)})")
                return result
        
        # Si la recherche par pattern échoue, reconstruire à partir des tokens
        result = ' '.join(improved_tokens[start_pos:end_pos])
        print(f"⚠️ Filtrage par reconstruction de tokens: {len(result)} caractères")
        return result
    
    print("\n✨ Aucun filtrage nécessaire")
    return improved_text

def main():
    # Charger les fichiers de test
    original_file = "/Users/qtresch/compare_transcription/output/videos/transcript_formatted_manifest [manifest] copie.txt"
    improved_file = "/Users/qtresch/compare_transcription/output/improved_transcripts/improved_manifest [manifest] copie.txt"
    
    try:
        with open(original_file, 'r', encoding='utf-8') as f:
            original_text = f.read()
        
        with open(improved_file, 'r', encoding='utf-8') as f:
            improved_text = f.read()
        
        print("=== TEST NOUVELLES RÈGLES DE FILTRAGE ===")
        print("Fichier:", "manifest [manifest] copie")
        print("Règles: 90% correspondance sur 30 tokens (début), arrêt à 30 tokens non correspondants (fin)")
        print("=" * 80)
        
        filtered_text = filter_transcription_segments(original_text, improved_text)
        
        print(f"\n{'='*80}")
        print("RÉSUMÉ FINAL:")
        print(f"{'='*80}")
        print(f"Longueur originale: {len(improved_text):,} caractères")
        print(f"Longueur filtrée: {len(filtered_text):,} caractères")
        print(f"Réduction: {len(improved_text) - len(filtered_text):,} caractères ({((len(improved_text) - len(filtered_text)) / len(improved_text) * 100):.1f}%)")
        
        if len(filtered_text) < len(improved_text):
            print(f"\n📄 DÉBUT DU TEXTE FILTRÉ:")
            print("-" * 50)
            print(filtered_text[:300] + "..." if len(filtered_text) > 300 else filtered_text)
            
            print(f"\n📄 FIN DU TEXTE FILTRÉ:")
            print("-" * 50)
            print("..." + filtered_text[-300:] if len(filtered_text) > 300 else filtered_text)
        else:
            print("\n📄 Aucun filtrage appliqué")
    
    except FileNotFoundError as e:
        print(f"Erreur: Fichier non trouvé - {e}")
    except Exception as e:
        print(f"Erreur: {e}")

if __name__ == "__main__":
    main()