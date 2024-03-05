#Importiamo le dipendenze
import requests
import pandas as pd
import json
import torch
import urllib.request
from pyquery import PyQuery as pq
import networkx as nx

#Richiamo il modello
from sentence_transformers import SentenceTransformer
model_similarity = SentenceTransformer('paraphrase-distilroberta-base-v1')

def BestDescrizione(lista_descrizioni, query):
  #Calcolo la similarità tra le descrizioni e la query
  sentence1 = model_similarity.encode(lista_descrizioni, convert_to_tensor=True)
  sentence2 = model_similarity.encode([query], convert_to_tensor=True)

  #Ritorno la coseno_similarità tra la parola e le descrizioni
  cosine_similarity = torch.nn.functional.cosine_similarity(sentence1, sentence2)
  return cosine_similarity.cpu().detach().numpy().argmax()

def Sinonimi(query):
  #Ritorno i sinonimi della parola "query"
  len_sinonimi=5
  try:
    d = pq(url=f'https://www.thesaurus.com/browse/{query}')
    lista_sinonimi = d('a.css-1kg1yv8.eh475bn0').text().split()
  except:
    lista_sinonimi = []
  return lista_sinonimi[:len_sinonimi] if len(lista_sinonimi)>0 else None


# Definiamo l'indirizzo base
API_WIKI = "https://wikidata.org/w/api.php"


def SearchOnWikiSemantic(query):
  #Ricerca su wikidata
  if 'None' in query:
    return None

  params = {
    'action': 'wbsearchentities',
    'format': 'json',
    'language': 'en',
    'search': query
  }

  r = requests.get(API_WIKI, params=params).json()
  lista_descrizioni = [r['search'][i]['description'] if 'description' in r['search'][i] else 'None' for i in
                       range(len(r['search']))]

  if len(lista_descrizioni) > 1:
    # Funzione che mi ritorna la posizione della descrizione migliore
    PosizioneBestDescrizione = BestDescrizione(lista_descrizioni, query)
    return lista_descrizioni[PosizioneBestDescrizione]

  return None if len(lista_descrizioni) == 0 else lista_descrizioni[0]


def remove_duplicates(lista):
  # Rimuove i duplicati dalla lista
  if not lista:
    return ['None']
  return list(dict.fromkeys(lista))

import re
def RemovePunct(text):
  #Rimuove le punct
  return re.sub(r'[^\w\s]', '', text) if text else 'None'

from gensim.parsing.preprocessing import remove_stopwords
#Funzione che rimuove le Stop Words
def RemoveStopWords(text):
  #Rimuove le stopwords
  return remove_stopwords(text) if text else 'None'


# importing modules
import nltk
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Init the Wordnet Lemmatizer
lemmatizer = WordNetLemmatizer()


def Lemma(sentence):
  # Tokenize: Split the sentence into words
  word_list = word_tokenize(sentence)
  # Lemmatize list of words and join
  lemmatized_output = [lemmatizer.lemmatize(w) for w in word_list]
  return lemmatized_output

def ConoscenzaBase(text_meme, caption):
  #Rimuovo caratteri non necessari
  #Punct
  text_pulito = RemovePunct(text_meme)
  #StopWords
  text_pulito = RemoveStopWords(text_pulito)
  #Stemming
  text_pulito = Lemma(text_pulito)
  #Rimuovo duplicati
  lista_parole_meme = remove_duplicates(text_pulito)

  #caption
  #Punct
  text_pulito_cap = RemovePunct(caption)
  #StopWords
  text_pulito_cap = RemoveStopWords(text_pulito_cap)
  #Tolgo la parola photo
  text_pulito_cap = text_pulito_cap.replace('photo', '')
  #Stemming
  text_pulito_cap = Lemma(text_pulito_cap)
  #Rimuovo duplicati
  lista_parole_cap = remove_duplicates(text_pulito_cap)


  return lista_parole_meme, lista_parole_cap

def ConoscenzaLivelloUno(list_words_meme):

  #Ad ogni parola presente nella lista trovo la corrispondente descrizione.
  ricerche_lvl_1=[]
  #Questa è informazione di livello 1
  ricerche_lvl_1 = [SearchOnWikiSemantic(elem) for elem in list_words_meme]
  #Contemporaneamente Rimuovo le StopWords
  ricerche_lvl_1 = [RemoveStopWords(elem) if elem else 'None' for elem in ricerche_lvl_1]
  #Rimuovo punct
  ricerche_lvl_1 = [RemovePunct(elem) if elem else 'None' for elem in ricerche_lvl_1]

  #Cerco i sinonimi
  sinonimi_lvl_0 = [Sinonimi(elem) for elem in list_words_meme]

  #Faccio lemming delle ricerche di lvl_1
  list_words_meme_lvl_1 = [Lemma(elem) if elem else 'None' for elem in ricerche_lvl_1]
  #Per ogni sinonimo trovato devo splittare e ricercare su wikidata
  #->matrix of words
  list_words_meme_lvl_1 = [elem.split() if elem else 'None' for elem in ricerche_lvl_1]

  return list_words_meme_lvl_1, sinonimi_lvl_0

def ConoscenzaLivelloDue(list_words_meme_lvl_1):

  #Devo ora sviluppare la conoscenza di livello 2

  list_words_meme_lvl_2 = []
  #Ricerca informazioni di secondo livello
  for i in range(len(list_words_meme_lvl_1)):
    if not list_words_meme_lvl_1[i]:
      #Considero il caso in cui wikidata non ritorni nulla
      list_words_meme_lvl_2.append('None')
      lunghezza = 0
    else:
      lunghezza=len(list_words_meme_lvl_1[i])
    for j in range(lunghezza):
      #Effettuo qui la ricerca sulla lista di parole
      descrizione_lvl_2 = SearchOnWikiSemantic(list_words_meme_lvl_1[i][j])
      #Rimuovo le stopwords
      if descrizione_lvl_2 != 'None':
        descrizione_lvl_2 = RemoveStopWords(descrizione_lvl_2)
        #Aggiungo
        if descrizione_lvl_2 != '':
          list_words_meme_lvl_2.append(descrizione_lvl_2)
        else:
          list_words_meme_lvl_2.append('None')
      else:
        list_words_meme_lvl_2.append('None')

  #Terminata la ricerca di informazioni devo splittare sui termini
  for i in range(len(list_words_meme_lvl_2)):
    if list_words_meme_lvl_2[i]:
      list_words_meme_lvl_2[i] = list_words_meme_lvl_2[i].split()

  return list_words_meme_lvl_2


def GenerazioneGrafo(list_words_meme, list_words_meme_lvl_1, list_words_meme_lvl_2, list_words_caption, sinonimi_lvl_0):
  # Genero ora la matrice di adiacenza che mi serve per generare il grafo

  # list_words_meme ---> contiene i termini di base del meme
  # sinonimi_lvl_0  ---> contiene i sinonimi di base al meme
  # list_words_meme_lvl_1
  # list_words_meme_lvl_2

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

