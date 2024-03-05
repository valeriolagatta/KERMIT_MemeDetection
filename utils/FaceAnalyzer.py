from deepface import DeepFace

# Questa funzione rileva gender, race ed emotion in un'immagine
def RilevazioneVolto(path):
    path_deepface = 'path_immagini'

    lista_rilevazione = []

    try:
        obj = DeepFace.analyze(img_path=path_deepface, actions=['gender', 'race', 'emotion'], prog_bar=False)
    except:
        #Se non rilevo volti
        obj = {
            "gender": None,
            "dominant_race": None,
            "dominant_emotion": None
        }

    lista_rilevazione = [obj['gender'], obj['dominant_race'], obj['dominant_emotion']]

    return lista_rilevazione
