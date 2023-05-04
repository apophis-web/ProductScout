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
import hashlib

df = pd.read_csv('processed_data.csv')
dataset = pd.read_csv("../Model-Training/dataset.csv")
dataset['Price'] = dataset['Price'].str.replace('Rs. ', '').str.replace(',', '')
dataset['Price'] = dataset['Price'].astype(float)
dataset['Date'] = pd.to_datetime(dataset['Date'])
max_date = dataset['Date'].max()
start_date = max_date - pd.DateOffset(days=7)
dataset = dataset[(dataset['Date'] >= start_date) & (dataset['Date'] <= max_date)]
grouped = dataset.groupby(['Category', 'Sub-Category', 'SubSub-Category'])
result = grouped['Price'].agg(['max', 'min', 'mean'])
result = result.reset_index()
result['mean'] = result['mean'].apply(lambda x: round(x, 0))
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
client = MongoClient("mongodb://hassan:1234@ac-a5jnqbt-shard-00-00.9arxrak.mongodb.net:27017,ac-a5jnqbt-shard-00-01.9arxrak.mongodb.net:27017,ac-a5jnqbt-shard-00-02.9arxrak.mongodb.net:27017/test?replicaSet=atlas-ckjaji-shard-0&ssl=true&authSource=admin")
db = client.scoutuser
users = db['users']

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
    
    days_s = 30
    rate_of_change = getpred(filtered_df, "Rate of Change", days_s, days_s)
    standard_dev = getpred(filtered_df, "Standard Dev", days_s, days_s)
    popularity = getpred(filtered_df, "Popularity", days_s, days_s)
    popularity = [round(num) for num in popularity]

    last_row = df.iloc[-1]
    last_date = datetime.strptime(str(last_row['Year']) + str(last_row['Month']).zfill(2) + str(last_row['Day']).zfill(2), '%Y%m%d')
    days = []
    week_day = []
    for i in range(days_s):
        next_date = last_date + timedelta(days=i+1)
        next_day = next_date.strftime('%d')
        next_weekday = next_date.strftime('%A')
        days.append(next_day)
        week_day.append(next_weekday)
    days = [str(int(num)) for num in days]


    main_list = []
    for i in range(days_s):
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
        temp.append(popularity[i]) 
        #temp.append(0) #APPENDING 0 INSTEAD OF POPULARITY VALUES FOR THE TIME BEING
        main_list.append(temp)

    return {"mapping":categoric_mapping, "main_list":main_list}


@app.route('/get_all_categories', methods = ['GET'])    
def get_all_categories():
    return {"categories":[', '.join(sublist) for sublist in df[['Sub-Category', 'Sub-Sub-Category']].drop_duplicates().values.tolist()]}

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

    # for i in range(len(temp)):
    #     date = str(temp["Year"].tolist()[i]) + "-" + str(temp["Month"].tolist()[i]) + "-" + str(temp["Day"].tolist()[i]) + " (" + str(temp["Week_Day"].tolist()[i]) + ") "
    #     original.append({ 'x': date, 'label': temp["Label"].tolist()[i] })

    # last_date = str(temp["Year"].tolist()[len(temp)-1]) + "-" + \
    #             str(temp["Month"].tolist()[len(temp)-1]) + "-" + \
    #             str(temp["Day"].tolist()[len(temp)-1]) + " (" + \
    #             str(temp["Week_Day"].tolist()[len(temp)-1]) + ") "

    last_date = "2023-04-11 (Tuesday)"
    
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
        original.append({ 'x': next_dates[i], 'trend': predictions[i] })
    mean = sum(predictions) / len(predictions)
    trend = calculate_trend(predictions)

    
    latest_year = df['Year'].max()
    latest_month = df[df['Year'] == latest_year]['Month'].max()
    latest_day = df[(df['Year'] == latest_year) & (df['Month'] == latest_month)]['Day'].max()
    latest_data = df[(df['Year'] == latest_year) & (df['Month'] == latest_month) & (df['Day'] == latest_day)]
    latest_data = latest_data.reset_index(drop=True)


    dictionary = {}
    for i in range(len(latest_data)):
        mapping = latest_data["Category"].iloc[i] + " " + latest_data["Sub-Category"].iloc[i] + " " + latest_data["Sub-Sub-Category"].iloc[i]
        dictionary[mapping] = latest_data["Label"].iloc[i]

    selected_mapping = cat + " " + subcat #+ " " + subsubcat
    filtered_dict = {k: v for k, v in dictionary.items() if k.startswith(selected_mapping)}

    bar_chart_data = []
    for i in range(len(filtered_dict)):
        sub_dict = {}
        sub_dict['name'] = list(filtered_dict.keys())[i]
        sub_dict['value'] = list(filtered_dict.values())[i]
        bar_chart_data.append(sub_dict)

    score = 0
    if (threshold < mean):
        score += 3
        threshold_val = "High"
    elif ((threshold / 2) < mean):
        score += 2
        threshold_val = "Moderate"
    elif ((threshold / 2) > mean):
        score += 1
        threshold_val = "Low"
        
    if trend > 0:
        score += 2
        trend_val = "Increasing"
    else:
         score -= 1
         trend_val = "Decreasing"

    if(score == 5):
        tfs = "98%"
    elif(score == 4):
        tfs = "78%"
    elif(score == 3):
        tfs = "63%"
    elif(score == 2):
        tfs = "46%"
    elif(score == 1):
        tfs = "22%"
    max_price = list(result[(result['Category'] == cat) & (result['Sub-Category'] == subcat) & (result['SubSub-Category'] == subsubcat)]['max'])[0]
    min_price = list(result[(result['Category'] == cat) & (result['Sub-Category'] == subcat) & (result['SubSub-Category'] == subsubcat)]['min'])[0]
    mean_price = list(result[(result['Category'] == cat) & (result['Sub-Category'] == subcat) & (result['SubSub-Category'] == subsubcat)]['mean'])[0]

    return {"pred": pred, "original": original, "threshold": threshold_val, "trend": trend_val, 
            "min_price": min_price, "max_price": max_price, "mean_price": mean_price, "bar_chart_data":bar_chart_data, "score":tfs}


@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    username = data['username']
    password = data['password']

    if not username or not password:
        return jsonify({'error': 'Missing username or password'}), 200

    user = users.find_one({'username': username})

    if user:
        return jsonify({'error': 'Username already exists'}), 200

    hashed_password = hashlib.sha256(password.encode()).hexdigest()

    new_user = {
        'username': username,
        'password': hashed_password,
        'preference':[]
    }
    users.insert_one(new_user)

    return jsonify({'message': 'User created successfully'}), 200

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data['username']
    password = data['password']

    if not username or not password:
        return jsonify({'error': 'Missing username or password'}), 200

    user = users.find_one({'username': username})

    if not user:
        return jsonify({'error': 'User not found'}), 200

    hashed_password = hashlib.sha256(password.encode()).hexdigest()

    if user['password'] != hashed_password:
        return jsonify({'error': 'Invalid password'}), 200

    return jsonify({'message': 'Logged in successfully'}), 200

@app.route('/prefset', methods=['POST'])
def prefset():
    data = request.get_json()
    username = data['username']
    pref = data['pref']
    users.update_one({"username":username },
             {"$set" : {"preference":pref}})
    return jsonify({'message': 'Done'}), 200


@app.route('/get_all_pref', methods = ['GET'])    
def get_all_pref():
    return {"categories":[', '.join(sublist) for sublist in df[['Category']].drop_duplicates().values.tolist()]}


@app.route('/get_user_hot', methods = ['POST'])    
def get_user_hot():
    data = request.get_json()
    username = data['username']
    user = users.find_one({'username': username})

    cat = user['preference']
    allprod = []
    indup = {}
    for i in cat:
        temp = df[df['Category'] == i]
        for j in range(len(temp)):
            sub_sub_category = temp.iloc[j]['Sub-Sub-Category']
            rate_of_change = temp.iloc[j]['Label']
            if sub_sub_category not in indup:
                indup[sub_sub_category] = rate_of_change
            else:
                indup[sub_sub_category] = max(indup[sub_sub_category], rate_of_change)

    allprod = [{k: v} for k, v in indup.items()]
    sorted_data = sorted(allprod, key=lambda item: list(item.values())[0], reverse=True)
    grouped_df = df.groupby(['Year', 'Month', 'Day']).agg({'Rate of Change': 'mean'}).reset_index()
    x = list(grouped_df[-30:]['Rate of Change'])
    tremar = []
    avg = 0
    count = 0
    for i in range(len(x)):
        avg += x[i]
        count += 1
        tremar.append({"name":i, 'market_trend':x[i], 'running_average': avg/count})

    return {"hot":sorted_data[:10], 'tre':tremar}



if __name__ == "__main__":
    app.run(port = 5000,debug = True,)