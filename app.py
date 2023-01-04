from flask import Flask, render_template, request, url_for, redirect,session
from flask_sqlalchemy import SQLAlchemy
from py.Twitter import TakeTwitterData
from py.Preprocessing import PreprocessingText
from py.ModelRun import SentimenNB,TopicModeling

import os
import pickle
import json
import pandas as pd
from datetime import datetime

tweetScrap = TakeTwitterData()

#
#preProcess.processingTweet


app = Flask(__name__)
app = Flask(__name__,template_folder='temp')

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///DiaryKuliah.db'
db = SQLAlchemy(app)

class Topic(db.Model):
  id_topic = db.Column(db.Integer,db.Sequence('seq_reg_id', start=1, increment=1), primary_key = True)
  topic_name = db.Column(db.String(150), index = True, unique = True)
  preprocessing = db.Column(db.Boolean, default=False, nullable=False)
  emotion = db.Column(db.Boolean, default=False, nullable=False)
  modeling = db.Column(db.Boolean, default=False, nullable=False)
  lastUpdate = db.Column(db.DateTime, index = True, unique = False)
  numberAccess = db.Column(db.Integer, default=0)

  def __repr__(self):
    return '<Topic %r>' % self.topic_name

@app.route('/')
def index():
  topic = Topic.query.order_by(Topic.numberAccess.desc()).all()
  topic_dict = {}
  for no,t in enumerate(topic):
    topic_dict[no] = {
      "id":t.id_topic,
      "preprocessing":t.preprocessing,
      "emotion":t.emotion,
      "modeling":t.modeling,
      "topic":t.topic_name
    }
  return render_template("index.html",topic = topic_dict)

@app.route('/topik/<topik>/<whatMustOpen>')
def topik(topik,whatMustOpen):
  whatMustOpen = int(whatMustOpen)
  if whatMustOpen == 0:  
    df = pd.read_csv("database/topik/"+topik+"/DataFiks.csv")

    da = df.groupby(['Dominant_Topic','Topic_Keywords','emotion']).count().iloc[:,:1].unstack()
    da.columns = ["Netral", "Positif", "Negatif"]
    da = da.sort_values(by=['Positif'], ascending=[False]).reset_index()
    print(da.head())

    positif = df[df['emotion']==1]
    negatif = df[df['emotion']==2]
    netral  = df[df['emotion']==0]

    return render_template('content/topic.html',topik = topik,topikData = da.to_dict('index'), positif = positif.to_dict('index'), negatif = negatif.to_dict('index'), netral = netral.to_dict('index'))
  
  else:
    return render_template('content/topicProcess.html',topik = topik, whatMustOpen = whatMustOpen)

@app.route('/new-topic/<topik>',methods=['GET'])
def new_topic(topik):
  topik = topik.lower()
  top = Topic(topic_name=topik,lastUpdate = datetime.now())
  db.session.add(top)
  db.session.commit()

  try:
    os.mkdir("database/topik/"+topik)
    os.mkdir("database/topik/"+topik+"/mentah")
  except:
    pass

  tweetScrap.UpdateTweet(topik,'database/topik/'+topik)
  print("Preprocessing Start")
  PreprocessingText('database/topik/'+topik).processingTweet()
  top = Topic.query.filter(Topic.topic_name == topik).first()
  top.preprocessing = True
  top.lastUpdate    = datetime.now()
  db.session.commit()

  print("Analisis Sentimen Start")
  SentimenNB().predict(topik)
  top = Topic.query.filter(Topic.topic_name == topik).first()
  top.emotion = True
  top.lastUpdate    = datetime.now()
  db.session.commit()

  print("Klasterisasi Start")
  TopicModeling(topik).categoriesTweet(topik)
  top = Topic.query.filter(Topic.topic_name == topik).first()
  top.modeling = True
  top.lastUpdate    = datetime.now()
  db.session.commit()

@app.route('/preprocessing-topic/<topik>',methods=['GET'])
def preprocessing_topic(topik):
  print("Preprocessing Start")
  PreprocessingText('database/topik/'+topik).processingTweet()
  top = Topic.query.filter(Topic.topic_name == topik).first()
  top.preprocessing = True
  top.lastUpdate    = datetime.now()
  db.session.commit()
  print("Done")
  return "1"

@app.route('/emotion-topic/<topik>',methods=['GET'])
def emotion_topic(topik):
  print("Analisis Sentimen Start")
  SentimenNB().predict(topik)
  top = Topic.query.filter(Topic.topic_name == topik).first()
  top.emotion = True
  top.lastUpdate    = datetime.now()
  db.session.commit()
  print("Done")
  return "1"

@app.route('/klasterisasi-topic/<topik>',methods=['GET'])
def klasterisasi_topic(topik):
  print("Klasterisasi Start")
  TopicModeling(topik).categoriesTweet(topik)
  top = Topic.query.filter(Topic.topic_name == topik).first()
  top.modeling = True
  top.lastUpdate    = datetime.now()
  db.session.commit()
  print("Done")
  return "1"

@app.route('/upAccess-topic/<topik>',methods=['GET'])
def upAccess_topic(topik):
  top = Topic.query.filter(Topic.topic_name == topik).first()
  top.numberAccess += 1
  db.session.commit()


if __name__=='__main__':
  app.run()