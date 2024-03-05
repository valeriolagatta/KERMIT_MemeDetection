#Importiamo le dipendenze
import requests
import pandas as pd
import json
import torch
import urllib.request
from pyquery import PyQuery as pq
import networkx as nx

#Richiamo il modello per calcolare la similarity
from sentence_transformers import SentenceTransformer
model_similarity = SentenceTransformer('paraphrase-distilroberta-base-v1')

def BestDescrizione(lista_descrizioni, text_caption, k):
  #Calcolo la similarità tra le descrizioni(surfaceText) e la caption/testo
  #k = iperparametro che mi dice quanti relazioni in conceptnet devo prendere per ogni parola
  sentence1 = model_similarity.encode(lista_descrizioni, convert_to_tensor=True)
  sentence2 = model_similarity.encode([text_caption], convert_to_tensor=True)
  #Ritorno una lista con l'indice dei 3 elementi più simili
  lista_piu_simili = []
  cosine_similarity = torch.nn.functional.cosine_similarity(sentence1, sentence2)
  array = cosine_similarity.cpu().detach().numpy().tolist()
  for i in range(0,k): #setta quante descrizioni vuoi
    if len(array)==0:
      return None
    massimo = max(array)
    indice = array.index(massimo)
    lista_piu_simili.append(indice)
    array.pop(indice)
  
  return lista_piu_simili


def SearchOnConceptNet(query, text_caption, k):
  #k = iperparametro che mi dice quanti relazioni in conceptnet devo prendere per ogni parola
  #Ricerca in ConceptNet
  if 'None' in query:
    return None
  
  request =  'http://api.conceptnet.io/c/en/' + query + '?filter=/c/en&limit=10000' 

  obj = requests.get(request).json()
  lista_descrizioni = []
  for i in obj['edges']:
    if (i['surfaceText'] != None) and ('is a translation of' not in i['surfaceText']):
        text = str(i['surfaceText']).replace('[','').replace(']','')
        text = text.lower() #deve essere in lower case
        lista_descrizioni.append(text)
        

  if len(lista_descrizioni) > 1:
    # Funzione che mi ritorna gli elementi migliori
    lista_indici_descrizioni_migliori = BestDescrizione(lista_descrizioni, text_caption, k)
    descrizioni_migliori = []
    if lista_indici_descrizioni_migliori == None:
      return None
    for i in lista_indici_descrizioni_migliori:
        descrizioni_migliori.append(lista_descrizioni[i])
    return descrizioni_migliori
  else:
    return None  

#rimuoviamo duplicati, stopwords, punteggiatura e lemmatizziamo
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

def ConoscenzaLivelloUno(list_words_meme, text_caption, k):
  #k = iperparametro che mi dice quanti relazioni in conceptnet devo prendere per ogni parola
  #Ad ogni parola presente nella lista trovo la corrispondente descrizione.
  ricerche_lvl_1=[]
  #Questa è informazione di livello 1
  for word in list_words_meme:
    descrizioni = SearchOnConceptNet(word, text_caption, k)
    #gestiamo il caso in cui conceptnet non ritorni nulla
    if descrizioni == None:
      descrizioni = ['None', 'None', 'None', 'None', 'None']
    ricerche_lvl_1.append(descrizioni)
  
  #mi salvo le 5 descrizioni che devo ritornare e salvarle nel dataframe per ogni word
  top5 = ricerche_lvl_1

  #print(ricerche_lvl_1)
  #Contemporaneamente Rimuovo le StopWords
  temp = []
  for i in ricerche_lvl_1:
    temp2 = []
    for j in i:
      if j:
        elem = RemoveStopWords(j)
        temp2.append(elem)
      else:
        temp2.append('None')
    temp.append(temp2)
  ricerche_lvl_1 = temp
  #print(ricerche_lvl_1)
  #Rimuovo punct(Con ConceptNet non serve perchè le descrizioni non hanno punteggiatura)
  temp = []
  for i in ricerche_lvl_1:
    temp2 = []
    for j in i:
      if j:
        elem = RemovePunct(j)
        temp2.append(elem)
      else:
        temp2.append('None')
    temp.append(temp2)
  ricerche_lvl_1 = temp
  #print(ricerche_lvl_1)
  
  #Faccio lemming delle ricerche di lvl_1
  temp = []
  for i in ricerche_lvl_1:
    temp2 = []
    for j in i:
      if j:
        elem = Lemma(j)
        temp2.append(elem)
      else:
        temp2.append('None')
    temp.append(temp2)
  list_words_meme_lvl_1 = temp
  #print(list_words_meme_lvl_1)
  #Per ogni sinonimo trovato devo splittare e ricercare su wikidata
  #->matrix of words
  temp = []
  for i in list_words_meme_lvl_1:
    temp2 = []
    for j in i:
      for m in j:
        if m:
          elem = m.split()
          temp2.append(elem)
        else:
          temp2.append('None')
    temp.append(temp2)
  list_words_meme_lvl_1 = temp
  #print(list_words_meme_lvl_1)
  #ogni parola finisce in una lista a se, io devo mettere parole relative ad una stesso oggetto del meme tutte in una lista
  temp = []
  for i in list_words_meme_lvl_1:
    temp2 = []
    for j in i:
      temp2.append(j[0]) #devi prendere la stringa non una lista con la stringa
    temp.append(temp2)
  list_words_meme_lvl_1 = temp
  #print(list_words_meme_lvl_1)

  #devo rimuovere i duplicati
  temp = []
  
  for i in list_words_meme_lvl_1:
    temp2 = []
    for j in i:
      if j not in temp2:
        temp2.append(j)
    temp.append(temp2)
  list_words_meme_lvl_1 = temp

  #l'output sarà una lista di liste, dove la lista i-esima ha le parole relative all'oggetto i-esimo estratto dal testo/immagine del meme
  return list_words_meme_lvl_1, top5

def ConoscenzaLivelloDue(list_words_meme_lvl_1, text_caption, k):

  #Devo ora sviluppare la conoscenza di livello 2

  list_words_meme_lvl_2 = []
  #Ricerca informazioni di secondo livello
  for i in range(len(list_words_meme_lvl_1)):
    list_words_meme_prov = [] #deve essere una lista di liste di liste
    if not list_words_meme_lvl_1[i]:
      #Considero il caso in cui conceptnet non ritorni nulla per la parola i-esima
      list_words_meme_prov.append('None')
      lunghezza = 0
    else:
      lunghezza=len(list_words_meme_lvl_1[i]) #lunghezza della lista di parole della parola i-esima
    for j in range(lunghezza):
      #per ogni parola nella lista della parola i-esima faccio ricerca in conceptnet, e mi ritorna una lista di k elementi
      lista_descrizioni_lvl_2 = []
      lista_descrizioni_lvl_2 = SearchOnConceptNet(list_words_meme_lvl_1[i][j], text_caption, k)
      #se ConceptNet non mi ritorna nulla, la lista sarà vuota
      if lista_descrizioni_lvl_2 == None:
        lista_descrizioni_lvl_2 = ['None', 'None', 'None', 'None', 'None']
      #print(lista_descrizioni_lvl_2)

      list_words_meme_prov.append(lista_descrizioni_lvl_2)
    
    list_words_meme_lvl_2.append(list_words_meme_prov)

  top5 = list_words_meme_lvl_2

  return top5
