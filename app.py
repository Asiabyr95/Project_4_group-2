from flask import Flask, jsonify, render_template, request, redirect
import pandas as pd
import numpy as np
from modelHelper import ModelHelper

#################################################
# Flask Setup
#################################################
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

modelHelper = ModelHelper()

#################################################
# Flask Routes
#################################################

# HTML ROUTES - HOME
@app.route("/")
def home():
    return render_template("home.html")

# HTML ROUTES - TABLEAU1
@app.route("/tableau1")
def tableau1():
    return render_template("tableau1.html")

# HTML ROUTES - TABLEAU2
@app.route("/tableau2")
def tableau2():
    return render_template("tableau2.html")

# HTML ROUTES - MODEL
@app.route("/model")
def model():
    return render_template("model.html")

# HTML ROUTES - Report 
@app.route("/report")
def report():
    return render_template("report.html")

# HTML ROUTES - ABOUT US 
@app.route("/about_us")
def about_us():
    return render_template("about_us.html")

# HTML ROUTES - SOURCES
@app.route("/sources")
def sources():
    return render_template("sources.html")

@app.route("/predictions", methods=["POST"])
def predictions():
    content = request.json["data"]
    print(content)

    # parse
    grams = content["grams"]
    quality = content['quality']
    successful_transactions = content ['successful_transactions']
    rating = content ['rating']
    product_title_sentiment = content ['product_title_sentiment']

    
    preds = modelHelper.predictions(grams, quality, successful_transactions, rating, product_title_sentiment)

    return(jsonify({"ok": True, "prediction": str(preds)}))

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    return r

# Run the App
if __name__ == '__main__':
    app.run(debug=True)