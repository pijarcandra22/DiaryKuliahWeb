import re
import pickle
import pandas as pd
import numpy as np
from py.Preprocessing import PreprocessingText
from py.NaiveBayes import MultiNB
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel

class SentimenNB:
  def __init__(self):
    self.model = None
    self.corpus = None
    try:
      self.model = pickle.load(open('py/naiveBayes.sav', 'rb'))
    except:
      self.train()
  
  def predict(self,topik):
    df = pd.read_csv('database/topik/'+topik+'/DataClean.csv')
    vectorizer = CountVectorizer(max_features= 2500, min_df = 1, max_df = 0.1)
    X_tweet = vectorizer.fit_transform(df['cleanTweet'])

    df['emotion'] = self.model.predict(X_tweet.toarray())
    df.to_csv('database/topik/'+topik+'/DataEmotion.csv')

  def train(self):
    preProcess = PreprocessingText()
    preProcess.processingTweet()

    train = pd.read_csv("database/training/DataClean.csv")
    train = train.replace("negative","2")
    train = train.replace("positive","1")
    train = train.replace("neutral","0")
    train = train.replace("netral","0")

    target_grow    = train['emotion'].value_counts()[0]
    target_change  = train['emotion'].value_counts().index.tolist()

    newForm = pd.DataFrame(columns=["emotion", "text", "cleanTweet"])
    for emo in range(len(target_change)-1):
      print(target_change[emo])
      temp = train[train['emotion']==target_change[emo]].copy()
      if(emo != 0):
        temp = temp.sample(target_grow, replace=True)
      newForm = pd.concat([newForm,temp],axis=0)

    print(len(newForm))
    train = newForm

    X = train['cleanTweet']
    Y = train['emotion']

    lb = LabelEncoder()
    cv = CountVectorizer()

    y_scratch = lb.fit_transform(Y)
    x_scratch = cv.fit_transform(X).toarray()

    X_train, X_test, y_train, y_test = train_test_split(x_scratch,y_scratch)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    model_scartch = MultiNB(alpha=0.001)
    model_scartch.fit(X_train, y_train)
    model_scartch.fit(X_train, y_train)
    yhat = model_scartch.predict(X_test)
    ytra = model_scartch.predict(X_train)
    print("Akurasi Test  = ",accuracy_score(y_test,yhat))
    print("Presisi Test  = ",precision_score(y_test,yhat,average='weighted'))
    print("Recall Test   = ",recall_score(y_test,yhat,average='macro'))
    print("="*30)
    print("Akurasi Train = ",accuracy_score(y_train,ytra))
    print("Presisi Train = ",precision_score(y_train,ytra,average='weighted'))
    print("Recall Train  = ",recall_score(y_train,ytra,average='macro'))

    pickle.dump(model_scartch, open("py/naiveBayes.sav", 'wb'))
    self.model = pickle.load(open('py/naiveBayes.sav', 'rb'))

class TopicModeling:
  def __init__(self,topik):
    self.LdaModel = None
    self.buildClass(topik)

  def calculate_topic_coherence(self,n_topics, alpha, eta, id_to_word, text, gens_corpus):
    gens_lda_multi = LdaMulticore(
                            corpus = gens_corpus,
                            id2word = id_to_word,
                            num_topics = n_topics,
                            chunksize = 5000,
                            passes = 20,
                            alpha = alpha,
                            eta = eta,
                            workers = 3,
                            random_state = 17
                            )
    
    gens_coherence_lda = CoherenceModel(
                            model = gens_lda_multi, 
                            texts = text, 
                            dictionary = id_to_word, 
                            coherence = 'c_v'
                            )
    
    gens_lda_multi_perplexity = gens_lda_multi.log_perplexity(gens_corpus)
    perplexity = np.exp2(-1.0 * gens_lda_multi_perplexity)
    
    return gens_coherence_lda.get_coherence(), perplexity,gens_lda_multi

  def format_topics_sentences(self,fulldf):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(self.LdaModel[self.corpus]):
        row = row_list[0] if self.LdaModel.per_word_topics else row_list
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = self.LdaModel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    sent_topics_df = pd.concat([fulldf,sent_topics_df], axis=1)
    return(sent_topics_df)

  def buildClass(self,topik):
    df = pd.read_csv('database/topik/'+topik+'/DataClean.csv')
    df['cleanTweet_list'] = df['cleanTweet'].apply(lambda x:x.split())

    data = df.cleanTweet_list.values.tolist()
    id2word = Dictionary(data)

    id2word.filter_extremes(no_below=2, no_above=.99)
    self.corpus = [id2word.doc2bow(d) for d in data]

    range_n_topics = np.arange(10,30,10)
    range_alpha = list(np.arange(0.1, 0.4, 0.3))
    range_eta = list(np.arange(0.1, 0.4, 0.3)) 
    range_alpha.append('symmetric')
    range_eta.append('symmetric')

    params_dict = {'modelLDA':[] ,'num_topics': [], 'alpha': [],'eta': [], 'coherence_score': [], 'perplexity': [] }

    num_cycles = len(range_n_topics)*len(range_alpha)*len(range_eta)
    count_cycles = 0

    for n_topics in range_n_topics:
        for alpha in range_alpha:
            for eta in range_eta:

                # print statements to check the status of execution
                count_cycles+=1
                print(f'Calculating:\nnum topics = {n_topics}\nalpha = {alpha}\neta = {eta}\nCurrent cycle: {count_cycles}/{num_cycles}')

                topic_coherence, perplexity,modelLDA = self.calculate_topic_coherence(
                                                            n_topics,
                                                            alpha,
                                                            eta,
                                                            id2word,
                                                            data,
                                                            self.corpus
                                                        )
                params_dict['modelLDA'].append(modelLDA)
                params_dict['perplexity'].append(perplexity)   
                params_dict['coherence_score'].append(topic_coherence)                
                params_dict['num_topics'].append(n_topics)
                params_dict['alpha'].append(alpha)
                params_dict['eta'].append(eta)

                # print statement to show that cycle is over 
                print(f'\nCoherence score calculated: {topic_coherence}')
                print(f'\nPerplexity score calculated: {perplexity}')
                print('\n\t\t# # # # # # # # # # # #\n')

    parameter_corpus = pd.DataFrame.from_dict(params_dict)
    param_model = parameter_corpus.sort_values(by=['coherence_score','perplexity'], ascending=[False,True]).reset_index(drop=True)
    param_model.to_csv('database/topik/'+topik+'/ParamCorpus.csv')
    pickle.dump(param_model.iloc[0,:]['modelLDA'], open('database/topik/'+topik+'/modelLDA.sav', 'wb'))
    self.LdaModel = pickle.load(open('database/topik/'+topik+'/modelLDA.sav', 'rb'))

    the_topic = {'idx': [], 'topic': []}
    for idx, topic in param_model.iloc[0,:]['modelLDA'].print_topics(-1):
        the_topic['idx'].append(idx)
        d1 = re.sub(r'[^a-zA-Z_+]','', topic)
        d1 = d1.replace('_',' ')
        d1 = d1.replace('+',', ')
        the_topic['topic'].append(d1)

    coherence_model = CoherenceModel(model=param_model.iloc[0,:]['modelLDA'], texts=data, 
                                      dictionary=id2word, coherence='c_v')
    lda_best_model_coherence = coherence_model.get_coherence_per_topic()
    the_topic['Coherence per Topik'] = lda_best_model_coherence
    dataTopik = pd.DataFrame.from_dict(the_topic)

    dataTopik.to_excel('database/topik/'+topik+'/DataClean.xlsx',index = False, float_format="%.3f")
  
  def categoriesTweet(self,topik):
    df = pd.read_csv('database/topik/'+topik+'/DataEmotion.csv')
    dataSiap = self.format_topics_sentences(df[['text','cleanTweet','emotion']])
    dataSiap.to_csv('database/topik/'+topik+'/DataFiks.csv')
    print("Done")