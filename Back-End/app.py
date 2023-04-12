import pickle
import math
import numpy as np
from pymongo import MongoClient
from flask_cors import CORS, cross_origin
from flask import Flask, request, jsonify
from datetime import datetime
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import statsmodels.api as sm
from datetime import datetime, timedelta

df = pd.read_csv('processed_data.csv')
threshold = df['Label'].mean()
threshold = round(threshold, 3)

with open(r'model.pkl', 'rb') as f:
    model = pickle.load(f)  

with open(r'cat_encoder.pkl', 'rb') as f:
    cat_encoder = pickle.load(f)

with open(r'sub_cat_encoder.pkl', 'rb') as f:
    sub_cat_encoder = pickle.load(f) 

with open(r'subsub_cat_encoder.pkl', 'rb') as f:
    subsub_cat_encoder = pickle.load(f)

with open(r'categories.pkl', 'rb') as f:
    categories = pickle.load(f)

with open(r'category_embeddings_categorical.pkl', 'rb') as f:
    category_query_embeddings_categorical = pickle.load(f)

with open(r'category_embeddings_hierarchical.pkl', 'rb') as f:
    category_query_embeddings_hierarchical = pickle.load(f)

# sentence_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v1')
sentencetransformer = SentenceTransformer('bert-base-nli-mean-tokens')

def getpred(df, columnname, days, span = 7):
    ema_20 = df[columnname].ewm(span=span).mean()  
    model = sm.tsa.statespace.SARIMAX(ema_20, order=(1,0,1), seasonal_order=(0,0,0,0))
    results = model.fit()
    forecast = results.predict(start=len(ema_20), end=len(ema_20)+4)
    forecast = results.predict(start=len(ema_20), end=len(ema_20)+days-1)
    return list(forecast)

def calculate_trend(values):
    x = np.arange(len(values))
    y = np.array(values)
    a, b = np.polyfit(x, y, 1)
    return a

app = Flask(__name__)
cors = CORS(app)

@app.route('/search', methods = ['POST'])
def get_cat():
    r = request.json
    user_query = r["name"]

    category_scores = []
    user_query_embedding = sentencetransformer.encode([user_query])[0]
    for i, category in enumerate(categories):
        score = cosine_similarity(user_query_embedding.reshape(1, -1), category_query_embeddings_categorical[i].reshape(1, -1))
        category_scores.append((category, score))
    sorted_scores_categorical = sorted(category_scores, key=lambda x: x[1], reverse=True)

    category_scores_hierarchical = []
    user_query_embedding = sentencetransformer.encode([user_query])[0]
    for i, category in enumerate(categories):
        category1_score = cosine_similarity(user_query_embedding.reshape(1, -1), category_query_embeddings_hierarchical[i].reshape(1, -1))
        category2_score = cosine_similarity(user_query_embedding.reshape(1, -1), category_query_embeddings_hierarchical[i+len(categories)].reshape(1, -1))
        category3_score = cosine_similarity(user_query_embedding.reshape(1, -1), category_query_embeddings_hierarchical[i+len(categories)*2].reshape(1, -1))
        score = (category1_score + category2_score + category3_score) / 3
        category_scores_hierarchical.append((category, score))

    sorted_scores_hierarchical = sorted(category_scores_hierarchical, key=lambda x: x[1], reverse=True)

    SCORES1 = []
    SCORES1.append(sorted_scores_categorical[0][0])
    SCORES1.append(sorted_scores_categorical[1][0])
    SCORES1.append(sorted_scores_categorical[2][0])

    SCORES2 = []
    SCORES2.append(sorted_scores_hierarchical[0][0])
    SCORES2.append(sorted_scores_hierarchical[1][0])
    SCORES2.append(sorted_scores_hierarchical[2][0])

    combined_list = SCORES1 + SCORES2
    counts = {}
    for lst in combined_list:
        key = tuple(lst)
        if key in counts:
            counts[key] += 1
        else:
            counts[key] = 1

    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    for key, count in sorted_counts:
        if list(key) in SCORES1 or list(key) in SCORES2:
            categoric_mapping = list(key)
            break


    filtered_df = df[(df["Category"] == categoric_mapping[0]) & (df["Sub-Category"] == categoric_mapping[1]) & (df["Sub-Sub-Category"] == categoric_mapping[2])]
    
    rate_of_change = getpred(filtered_df, "Rate of Change", 7, 7)
    standard_dev = getpred(filtered_df, "Standard Dev", 7, 7)
    popularity = getpred(filtered_df, "Popularity", 7, 7)
    popularity = [round(num) for num in popularity]

    last_row = df.iloc[-1]
    last_date = datetime.strptime(str(last_row['Year']) + str(last_row['Month']).zfill(2) + str(last_row['Day']).zfill(2), '%Y%m%d')
    days = []
    week_day = []
    for i in range(7):
        next_date = last_date + timedelta(days=i+1)
        next_day = next_date.strftime('%d')
        next_weekday = next_date.strftime('%A')
        days.append(next_day)
        week_day.append(next_weekday)
    days = [str(int(num)) for num in days]


    main_list = []
    for i in range(7):
        temp = []
        temp.append(categoric_mapping[0])
        temp.append(categoric_mapping[1])
        temp.append(categoric_mapping[2])
        temp.append(2023)
        temp.append(4)
        temp.append(int(days[i]))
        temp.append(week_day[i])
        temp.append(rate_of_change[i])
        temp.append(standard_dev[i])
        #temp.append(popularity[i]) 
        temp.append(0) #APPENDING 0 INSTEAD OF POPULARITY VALUES FOR THE TIME BEING
        main_list.append(temp)

    return {"mapping":categoric_mapping, "main_list":main_list}


@app.route('/get_text', methods = ['POST'])    
def get_text():
    r = request.json
    lists = r["lists"]
    cat = lists[0][0]
    subcat = lists[0][1]
    subsubcat = lists[0][2]

    weekday_encoding = {
        "Monday" : 1,
        "Tuesday" : 2,
        "Wednesday" : 3,
        "Thursday" : 4,
        "Friday" : 5,
        "Saturday" : 6,
        "Sunday" : 7
    }

    indices_to_transform = [0, 1, 2, 6]

    for lst in lists:
        for i in indices_to_transform:
            if i == 0:
                lst[i] = cat_encoder.transform([lst[i]])[0]
            elif i == 1:
                lst[i] = sub_cat_encoder.transform([lst[i]])[0]
            elif i == 2:
                lst[i] = subsub_cat_encoder.transform([lst[i]])[0]
            elif i == 6:
                lst[i] = weekday_encoding[lst[i]]

            

    predictions = []
    for i in range(len(lists)):
        predictions.append(model.predict([lists[i]]))
    predictions = [float(arr[0]) for arr in predictions]

    original = []
    pred = []

    count = 0
    temp = df[(df['Category'] == cat) & (df['Sub-Category'] == subcat) & (df['Sub-Sub-Category'] == subsubcat)]

    for i in range(len(temp)):
        date = str(temp["Year"].tolist()[i]) + "-" + str(temp["Month"].tolist()[i]) + "-" + str(temp["Day"].tolist()[i]) + " (" + str(temp["Week_Day"].tolist()[i]) + ") "
        original.append({ 'x': date, 'label': temp["Label"].tolist()[i] })

    last_date = str(temp["Year"].tolist()[len(temp)-1]) + "-" + \
                str(temp["Month"].tolist()[len(temp)-1]) + "-" + \
                str(temp["Day"].tolist()[len(temp)-1]) + " (" + \
                str(temp["Week_Day"].tolist()[len(temp)-1]) + ") "
    
    date_str = last_date.split(" ")[0]
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")

    next_dates = []
    for i in range(1, len(predictions) + 1):
        next_date_obj = date_obj + timedelta(days=i)
        next_date_str = next_date_obj.strftime("%Y-%m-%d")
        next_weekday_str = next_date_obj.strftime("%A")
        next_date_formatted = f"{next_date_str} ({next_weekday_str})"
        next_dates.append(next_date_formatted)


    for i in range(len(predictions)):
        original.append({ 'x': next_dates[i], 'label': predictions[i] })

    mean = sum(predictions) / len(predictions)
    trend = calculate_trend(predictions)


    if (threshold < mean):
        threshold_val = "High"
    elif ((threshold / 2) < mean):
        threshold_val = "Moderate"
    elif ((threshold / 2) > mean):
        threshold_val = "Low"
        
    if trend > 0:
        trend_val = "Increasing"
    else:
         trend_val = "Decreasing"

    return {"pred": pred, "original": original, "threshold": threshold_val, "trend": trend_val}
    
if __name__ == "__main__":
    app.run(port = 5000,debug = True,)