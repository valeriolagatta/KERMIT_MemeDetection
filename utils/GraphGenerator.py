import wikidata

def GenerazioneGrafo(list_words_meme, list_words_meme_lvl_1, list_words_meme_lvl_2, list_words_caption, sinonimi_lvl_0):
  # Genero ora la matrice di adiacenza che mi serve per generare il grafo

  # list_words_meme ---> contiene i termini di base del meme
  # sinonimi_lvl_0  ---> contiene i sinonimi di base al meme
  # list_words_meme_lvl_1 -> contiene la lista di conoscenza lvl 1
  # list_words_meme_lvl_2 -> contiene la lista di conoscenza lvl 2
  # list_words_caption -> contiene la lista di parole derivanti dal captioning

  rows = []

  # Partiamo col collegare totalmente le parole estratte dal meme
  for i in range(len(list_words_meme) - 1):
    rows.append([list_words_meme[i], list_words_meme[i + 1], 1])

  # Devo ora collegare ogni parola con la sua descrizione di livello 1
  for i in range(len(list_words_meme_lvl_1)):
    # Collego prima il grafo principale con quello di livello 1
    # Inserisco una profondità pari a 2, cioè sono al livello 1
    rows.append([list_words_meme[i], list_words_meme_lvl_1[i][0], 2])
    for j in range(len(list_words_meme_lvl_1[i]) - 1):
      rows.append([list_words_meme_lvl_1[i][j], list_words_meme_lvl_1[i][j + 1], 2])

  if list_words_meme_lvl_2:  # Se devo usare anche la conoscenza di livello 2
    # Devo a questo punto collegare le informazioni di livello 2 con quelle di lvl 1
    indice_lvl_2 = 0
    for i in range(len(list_words_meme_lvl_1)):
      for j in range(len(list_words_meme_lvl_1[i])):
        if (list_words_meme_lvl_1[i][j] or list_words_meme_lvl_2[indice_lvl_2][0]):
          rows.append([list_words_meme_lvl_1[i][j], list_words_meme_lvl_2[indice_lvl_2][0], 3])
          indice_lvl_2 += 1
        else:
          indice_lvl_2 += 1

    # Devo ora solo collegare tra loro le informazioni di livello 2
    for i in range(len(list_words_meme_lvl_2)):
      for j in range(len(list_words_meme_lvl_2[i]) - 1):
        # Se la lunghezza è pari ad uno, vai avanti
        if len(list_words_meme_lvl_2[i]) == 1:
          continue
        else:
          rows.append([list_words_meme_lvl_2[i][j], list_words_meme_lvl_2[i][j + 1], 3])

  # Collego i sinonimi
  for i in range(len(sinonimi_lvl_0)):
    if sinonimi_lvl_0[i]:
      for j in range(len(sinonimi_lvl_0[i])):
        rows.append([list_words_meme[i], sinonimi_lvl_0[i][j], 1])
    else:
      continue

  # Devo collegare le informazioni dell'immagine con i nodi principale
  for i in range(len(list_words_caption) - 1):
    # unisco prima l'ultimo col primo
    if i == 0:
      rows.append([list_words_meme[-1], list_words_caption[0], 1])
    else:
      rows.append([list_words_caption[i], list_words_caption[i + 1], 1])

  # Ho generato la matrice di adiacenza, ora deve diventare un grafo
  matrice_adiacenza_grafo_pd = pd.DataFrame(rows, columns=['source', 'target', 'depth'])
  graph = nx.from_pandas_edgelist(matrice_adiacenza_grafo_pd, source='source', target='target', edge_attr=True,
                                  create_using=nx.DiGraph())

  return matrice_adiacenza_grafo_pd, graph
