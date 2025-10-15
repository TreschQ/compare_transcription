#!/usr/bin/env python3
"""Script de debug détaillé pour comprendre le filtrage"""

import re

def filter_transcription_segments_debug(original_text, improved_text):
    """Version debug de la fonction de filtrage"""
    
    # Tokeniser les textes
    original_tokens = re.findall(r'\w+', original_text.lower())
    improved_tokens = re.findall(r'\w+', improved_text.lower())
    
    print(f"Tokens original: {len(original_tokens)}")
    print(f"Tokens amélioré: {len(improved_tokens)}")
    
    # 1. Trouver le point de départ
    start_pos = None
    original_start_30 = original_tokens[:30]
    
    for i in range(min(100, len(improved_tokens) - 29)):  # Chercher dans les 100 premiers
        improved_segment_30 = improved_tokens[i:i+30]
        matches = sum(1 for orig, imp in zip(original_start_30, improved_segment_30) if orig == imp)
        similarity = matches / 30
        
        if similarity >= 0.9:
            start_pos = i
            print(f"✅ Point de départ trouvé à position {i} avec {similarity:.1%} correspondance")
            break
    
    if start_pos is None:
        return improved_text
    
    # 2. Afficher les derniers tokens de l'original pour comprendre où ça doit finir
    print(f"\n30 derniers tokens originaux: {' '.join(original_tokens[-30:])}")
    
    # 3. Comparer token par token depuis le début
    original_idx = 0
    improved_idx = start_pos
    consecutive_mismatches = 0
    last_match_pos = start_pos
    
    print(f"\nComparaison token par token (affichage des dernières correspondances):")
    
    while improved_idx < len(improved_tokens) and original_idx < len(original_tokens):
        orig_token = original_tokens[original_idx]
        imp_token = improved_tokens[improved_idx]
        
        if orig_token == imp_token:
            consecutive_mismatches = 0
            last_match_pos = improved_idx
            # Afficher les 20 dernières correspondances
            if original_idx >= len(original_tokens) - 20:
                print(f"  Position {improved_idx}: '{imp_token}' = '{orig_token}' ✅ (original #{original_idx})")
            original_idx += 1
        else:
            consecutive_mismatches += 1
            if consecutive_mismatches <= 5:  # Afficher les premières non-correspondances
                print(f"  Position {improved_idx}: '{imp_token}' ≠ '{orig_token}' ❌ (consecutives: {consecutive_mismatches})")
            
            if consecutive_mismatches >= 30:
                end_pos = improved_idx - 29
                print(f"🛑 Arrêt à cause de 30 tokens non correspondants à position {improved_idx}")
                break
        
        improved_idx += 1
    
    # Vérifier si on a atteint la fin du texte original
    if original_idx >= len(original_tokens):
        print(f"📍 Fin du texte original atteinte à position improved_idx={improved_idx}, original_idx={original_idx}")
        print(f"📍 Dernière correspondance à position {last_match_pos}")
        end_pos = improved_idx
    else:
        end_pos = len(improved_tokens)
    
    print(f"\nRésultat: tokens[{start_pos}:{end_pos}] sur {len(improved_tokens)} total")
    
    # Montrer ce qui vient après la position de fin
    if end_pos < len(improved_tokens):
        print(f"\nTokens qui seraient supprimés (position {end_pos}+):")
        remaining_tokens = improved_tokens[end_pos:end_pos+20]
        print(f"  {' '.join(remaining_tokens)}...")
    
    return ' '.join(improved_tokens[start_pos:end_pos])

def main():
    # Charger les fichiers
    original_file = "output/videos/transcript_formatted_manifest [manifest] copie.txt"
    improved_file = "output/improved_transcripts/improved_manifest [manifest] copie.txt"
    
    with open(original_file, 'r', encoding='utf-8') as f:
        original_text = f.read()
    
    with open(improved_file, 'r', encoding='utf-8') as f:
        improved_text = f.read()
    
    print("=== DEBUG DÉTAILLÉ DU FILTRAGE ===")
    filtered_text = filter_transcription_segments_debug(original_text, improved_text)
    
    print(f"\n{'='*50}")
    print(f"Longueur originale: {len(improved_text):,}")
    print(f"Longueur filtrée: {len(filtered_text):,}")
    print(f"Réduction: {len(improved_text) - len(filtered_text):,}")

if __name__ == "__main__":
    main()