import pandas as pd
import numpy as np
import emoji
import re
import os
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

class Preprocessing:
    def __init__(self):
        self.factory   = StemmerFactory()
        self.stemmer   = self.factory.create_stemmer()
        self.stopWords = self.getStopWordList()

    def give_emoji_free_text(self,text):
        """Menghilangkan Emoji Pada Tweet"""
        emoji_list = [c for c in text if c in emoji.UNICODE_EMOJI]
        clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])
        return clean_text

    def url_free_text(self,text):
        """Menghilangkan Url Tweet"""
        text = re.sub(r'http\S+', '', text)
        return text

    def username_free_text(self,text):
        """Menghilangkan Username User"""
        result = re.sub(r'@\S+','', text)
        v = re.sub(r"^b'RT|^b'",'', result)
        return v
    
    def replaceTwoOrMore(self,s):
        """Menghilangkan Karakter Berulang"""
        pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
        return pattern.sub(r"\1\1", s)

    def getStopWordList(self):
        """Mengambil Stopword dari Library Sastrawi"""
        stopWords = StopWordRemoverFactory()
        more_stopword = ['dengan', 'ia','bahwa','oleh','AT_USER','URL','di','yg','dari','ke','ini','bgmn','tmn2','dr','pt','dg','prn','bn','sbb', 'tdk', 'krn', 'ga,',
                        'tak', 'gak', 'gk', 'bkn', 'kan', 'la', 'so', 'dgn', ]
        data = stopWords.get_stop_words()+more_stopword
        return data
    
    def steaming_text(self,sentence):
        """Stemming Pada Text"""
        return self.stemmer.stem(sentence)
    
    def getFeatureVector(self,tweet):
        """Melakukan Tokenisasi"""
        featureVector = []
        words = tweet.split()
        for w in words:
            w = self.replaceTwoOrMore(w)
            w = w.strip('\'"?,.').lower()
            val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
            if(w in self.stopWords or val is None):
                continue
            else:
                featureVector.append(w)
        return featureVector

class PreprocessingText(Preprocessing):
    def __init__(self,path=None):
        Preprocessing.__init__(self)
        """Data yang di Inputkan merupakan nama file csv tempat tweet disimpan"""
        self.path = None

        if path is None:
            self.csvFile = pd.DataFrame(columns=["emotion", "text"])
            for filename in os.listdir("database/training/mentah"):
                df_sementara   = pd.read_csv("database/training/mentah/"+filename, names = ["emotion", "text"])
                self.csvFile   = pd.concat([self.csvFile,df_sementara],axis=0)
            self.csvFile['emotion'] = self.csvFile['emotion'].str.replace(r'|','')
            self.csvFile['text'] = self.csvFile['text'].str.replace(r'|','')
            self.path = "database/training"
        else:
            self.csvFile = pd.DataFrame(columns=["datetime", "text"])
            for filename in os.listdir(path+"/mentah"):
                df_sementara   = pd.read_csv(path+"/mentah/"+filename, names = ["datetime", "text"])
                self.csvFile   = pd.concat([self.csvFile,df_sementara],axis=0)
            self.path = path
        self.csvFile = self.csvFile.drop_duplicates(subset=["text"])

    def processingTweet(self):
        call_emoji_free = lambda x: self.give_emoji_free_text(x)
        self.csvFile['cleanTweet'] = self.csvFile['text'].apply(call_emoji_free)
        self.csvFile['cleanTweet'] = self.csvFile['cleanTweet'].apply(self.url_free_text)
        self.csvFile['cleanTweet'] = self.csvFile['cleanTweet'].apply(self.username_free_text)
        self.csvFile['cleanTweet'] = self.csvFile['cleanTweet'].apply(self.getFeatureVector)
        self.csvFile['cleanTweet'] = [' '.join(map(str, l)) for l in self.csvFile['cleanTweet']]
        self.csvFile['cleanTweet'] = self.csvFile['cleanTweet'].apply(self.steaming_text)
        self.csvFile.dropna()
        self.csvFile = self.csvFile[self.csvFile.cleanTweet != '']
        try:
            df = pd.read_csv(self.path+"/DataClean.csv")
            self.csvFile = pd.concat([df,self.csvFile],axis=0)
        except:
            pass

        self.csvFile.to_csv(self.path+"/DataClean.csv",index=False)
        print("Preprocessing Selesai")
        for filename in os.listdir(self.path+"/mentah"):
            os.remove(self.path+"/mentah/"+filename)