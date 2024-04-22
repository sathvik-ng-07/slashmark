# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 17:45:40 2017

@author: NishitP
"""


import pickle

def detecting_fake_news(var):
    # Load the model from disk
    with open('final_model.sav', 'rb') as f:
        load_model = pickle.load(f)

    # Making prediction
    prediction = load_model.predict([var])
    prob = load_model.predict_proba([var])

    # Print results
    print("The given statement is:", "FAKE" if prediction[0] == 0 else "TRUE")
    print("The truth probability score is:", prob[0][1])

if __name__ == '__main__':
    var = input("Please enter the news text you want to verify: ")
    print("You entered:", var)
    detecting_fake_news(var)
