#!/bin/bash

# Choix de la taille initiale via une boîte de sélection
SIZE=$(zenity --list \
    --title="Choix de la taille initiale de la tumeur" \
    --radiolist \
    --column="Choisir" --column="Taille" \
    TRUE "petite" FALSE "normale" FALSE "grosse" 2>/dev/null)

# Si l'utilisateur annule
if [ -z "$SIZE" ]; then
    echo "Aucune sélection faite."
    exit 1
fi

# Demande du numéro de carte
CARD=$(zenity --entry \
    --title="Numéro de carte" \
    --text="Entre un numéro de carte (1 à 18):" 2>/dev/null)

# Vérification basique
if ! [[ "$CARD" =~ ^[0-9]+$ ]] || [ "$CARD" -lt 1 ] || [ "$CARD" -gt 18 ]; then
    zenity --error --text="Numéro de carte invalide (doit être entre 1 et 18)." 2>/dev/null
    exit 1
fi

# Mapping taille -> préfixe numérique
case $SIZE in
    "petite") PREFIX="1" ;;
    "normale") PREFIX="10" ;;
    "grosse") PREFIX="100" ;;
esac

# Construction du nom de fichier
FILENAME="${PREFIX}_C${CARD}.py"

# Vérifie si le fichier existe
if [ ! -f "$FILENAME" ]; then
    zenity --error --text="Le fichier $FILENAME n'existe pas." 2>/dev/null
    exit 1
fi

# Lance le script Python
zenity --info --text="Lancement de $FILENAME..." 2>/dev/null
python3 "$FILENAME"