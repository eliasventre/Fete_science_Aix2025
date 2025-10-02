#!/bin/bash

# Vérifie que zenity est installé
if ! command -v zenity &> /dev/null
then
    echo "Zenity n'est pas installé. Installe-le avec : brew install zenity"
    exit 1
fi

# Choix de la taille initiale via une boîte de sélection
SIZE=$(zenity --list \
    --title="Simulation PKPD" \
    --text="Choisis une taille de tumeur initiale:" 2>/dev/null \
    --radiolist \
    --column="Choisir" --column="Taille" \
    TRUE "petite" FALSE "moyenne" FALSE "grosse" 2>/dev/null)

# Si l'utilisateur annule
if [ -z "$SIZE" ]; then
    echo "Aucune sélection faite !"
    exit 1
fi

# Demande du numéro de carte
CARD=$(zenity --entry \
    --title="Simulation PKPD" \
    --text="Entre un numéro de carte (1 à 19):" 2>/dev/null)

# Vérification basique
if ! [[ "$CARD" =~ ^[0-9]+$ ]] || [ "$CARD" -lt 1 ] || [ "$CARD" -gt 19 ]; then
    zenity --error --text="Numéro de carte invalide (doit être entre 1 et 19)." 2>/dev/null
    exit 1
fi

# Mapping taille -> préfixe numérique
case $SIZE in
    "petite") PREFIX="1" ;;
    "moyenne") PREFIX="10" ;;
    "grosse") PREFIX="100" ;;
esac

# Construction du nom de fichier
FILENAME="${PREFIX}_C${CARD}.py"

# Vérifie si le fichier existe
if [ ! -f "./$FILENAME" ]; then
    zenity --error --text="Le fichier ./$FILENAME n'existe pas." 2>/dev/null
    exit 1
fi

# Lance le script Python directement (plus de popup finale)
python3 "./$FILENAME"