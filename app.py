
from flask import Flask,render_template,request
from recommendation_model import *
import logging
app = Flask(__name__)


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])

@app.route("/")
def getInput():
    return render_template("index.html")

@app.route("/submit",methods=['POST'])
def recommendProducts():
    try:
        if request.method=='POST':
            name = request.form["Username"]
            top_n = int(request.form["top_n_recommendations"])
            rm = RecommendationModel()
            rmList=rm.top_n_recommendations(name,top_n=20)
            prediction = rm.top_n_positive_recommendations(rmList,top_n=top_n)
        # 1. get recommended products
        # 2. sentiment analysis
        # 3. final prediction
            logging.info(prediction)

        return render_template("submit.html",prediction = prediction)
    except KeyError as e:
        return render_template("submit.html",error="Sorry username doesn't exists!")

if __name__=="__main__":
    app.run()