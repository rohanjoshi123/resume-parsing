# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 14:32:54 2020

@author: Rohan joshi
"""

import pandas as pd
import psycopg2
import sys
import datefinder
import datetime
import numpy as np
from datetime import date
#from forms import UploadJD
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
from itertools import groupby
from geotext import GeoText
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
import json
import re
from sklearn import naive_bayes
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import datefinder
from geotext import GeoText
import pickle
import itertools
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from math import sqrt
import string


#nltk.download('popular')

def extract_info(skill_roster, resume_text, resume_name):
    resume_source = ' -NA- '
    if 'dice' in resume_name.lower():
        resume_source = 'DICE'
    elif 'naukri' in resume_name.lower():
        resume_source = "NAUKRI"
    elif 'monster' in resume_name.lower():
        resume_source = 'MONSTER'
    elif "indeed" in resume_name.lower():
        resume_source = 'INDEED'
    elif "linkedin" in resume_name.lower():
        resume_source = 'LINKEDIN'
    else:
        resume_source = 'OTHER'

    resume_text = resume_text.replace(" Ã¢â¬â " , " to ")
    resume_text = resume_text.replace("Ã¢â¬â" , " to ")
    resume_text = resume_text.replace(" '", " ")
    resume_text = resume_text.replace("'","")
    resume_text = resume_text.replace(" Ã¢â¬â¢ ","")
    resume_text = resume_text.replace("Ã¢â¬â¢"," ")
    resume_text = resume_text.replace("[","")
    resume_text = resume_text.replace("]","")
    resume_text = resume_text.replace("Ã¢â¬â"," - ")

    #################################################### EMAIL ID #################################################################################################
#    print(resume_text)
    email = ' -NA- '
    try:
        email = re.findall("[\w\.]+@[\w\.-]+", resume_text)
        email = email[0]
        email = list(set(email))
        emails = str(email)
        email = emails.replace('[', '')
        email = email.replace(']', '')
        email = email.replace("'", '')
        if email == '':
            email = ' -NA- '               # Get the email id of the candidate
    except:
        email = ' -NA- '

    ################################# PHONE NO ######################################################################################################################
    phoneno = ' -NA- '
    try:
#        phoneno = re.findall(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', resume_text)   # Extract the phone number possibilities using REGEX
        phoneno = re.findall(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})', resume_text)
        if phoneno != []:
            phoneno = phoneno[0]

    except:
        pass

    ################################# NAME ###########################################################################################################################
    name_str = ' NA '
    try:
        #        with open(resume_path,"r", encoding="utf-8") as text:
    	resume_splitlines = resume_text.splitlines()
    	for entry in resume_splitlines:
                tokens = nltk.word_tokenize(entry)
                tagged = nltk.pos_tag(tokens)                   # Finding the Part of Speech for each words

                groups = groupby(tagged, key=lambda x: x[1]) 	# Group by tags
                names = [[w for w,_ in words] for tag,words in groups if (tag=="NNP" or tag == "NN")]  	# get the sequence of Proper Nouns

                names = [" ".join(name) for name in names if len(name)>=1]
                if names != []:
                    name_str = names[0]
                    break                        #get the first sequence of Proper Nouns

    except:
        e = sys.exc_info()[0]
        print(e)
        name_str = ' -NA- '

    ############################################## ABSTRACT #########################################################################################################

    resumes = [resume for resume in os.listdir("./Resume/")]
    
    path='./Resume/'
    resume_text1=[]
    
    for resume in resumes:
        with open(path+resume,"r",encoding = "utf-8-sig") as f:
            resume_text1.append(f.read().lower().replace(u"\uf0b7", u"").replace(u"\xa0",u"").replace(u"\u200b",u"").replace("\t",""))
            
    resume_text_lines=[]
    
    for text in resume_text1:
        each_text = text.splitlines()
        each_text = [txt for txt in each_text if txt!='']
        resume_text_lines.append(each_text)
    
    
    corpus = ['Summary', 'Work','Experience Summary',  'History', 'Experience','Career History', 'Employment History' , 'Accomplishments', 'Certifications','Professional', "Tools",'Education', 'Educational', 'Skills', 'Information', 'Academic' , 'Career']
    corpus_len = len(corpus)
    possible=[]
    
    for each_text in resume_text_lines:
        for strings in each_text:
            if len(strings.split())<=5:
                if len(strings.split()) ==1:
                    if len(strings.split()[0])>1:
                        possible.append(strings)
                else:
                    possible.append(strings)
                    
    possible = corpus + possible
    heading = []
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(possible)
    
    for j in range(corpus_len):
        matrix = cosine_similarity(tfidf_matrix[j:j+1], tfidf_matrix)
        for number in matrix:
            continue
        for i in range(len(number)):
            if number[i] >0.5 :
                # print(i)
                heading.append((possible[i]))
    heading = list(set(heading))    
    corpus = corpus + heading
    corpus = list(set(corpus))
    for i in range(len(corpus)):
        corpus[i] = re.sub(r"[^a-zA-Z0-9]+", ' ', corpus[i])
            
    # fname = '119259_1601_1_WHJordanJr_Resume_Jan2020.txt'
    # path1='C:\\Users\\rohan\\Desktop\\Fractionable\\new\\docx\\'    
    
     
    # try:
    #     resume_text = open(path1+fname).read()
    # except:
    #     resume_text = open(path1 +fname, encoding = 'utf-8').read()
    
    resume_text = resume_text.replace('-', ' to ')
    resume_text = resume_text.replace('â€”', ' to ')
    resume_text = resume_text.replace('â€“', ' to ')
    resume_text = resume_text.replace('–', ' to ')   
    resume_text = resume_text.replace('(', '')   
    resume_text = resume_text.replace(')', '')   
    
     
    # resume_text = re.sub(r"[^a-zA-Z0-9]+", ' ', resume_text)
    resume_splitlines = resume_text.splitlines()
    resume_splitlines = [entry for entry in resume_splitlines if entry!= '']
    
    for i in range(len(resume_splitlines)):
        resume_splitlines[i] = re.sub(r"[^a-zA-Z0-9/-]+", ' ', resume_splitlines[i])
    
    resume_splitlines = [entry for entry in resume_splitlines if entry!= ' ']
    # print(resume_splitlines)
    
    ending_index = []
    
    result = []
    try:
        if "dice" in resume_name.lower():
            for lines in resume_splitlines:
                if 'Work History' in lines:
                    starting_index = resume_splitlines.index(lines)
                    # print(starting_index)
                if  'Education' in lines:    
                    ending_index.append(resume_splitlines.index(lines))
                if 'Skills'  in lines:    
                    ending_index.append(resume_splitlines.index(lines))        
            ending_index.sort()   
            # print(ending_index)

            for index in ending_index:
                if index > starting_index:
                    result.append(resume_splitlines[starting_index:index])
                    # print(resume_splitlines[starting_index:index])
                    break
            name_index = []
            name_split = name_str.split()
            name_split = name_split[0]
            # print(name_split)
            for lines in resume_splitlines:
                if "Contact Information" in lines:
                    # print("Found")
                    name_index.append(resume_splitlines.index(lines))
                    # print(name_index)
            if name_index[-1] != 0:
                resume_splitlines = resume_splitlines[0:name_index[-1]]
        
            # print(resume_splitlines)
                
        
        # result.remove(result[0])
        # result.remove(result[1])
        # dice_result = []
        # for i in range (len(result)):
        #     if w(i+1 != len(result)+1):
        #         dice_result[i] = result[i:i+1]
        # result = dice_result
    except:
        pass
    
   
        
    # if result == []:
    indices = []
    for entries in corpus:
        for entries1 in resume_splitlines:
            if entries.lower() == entries1.lower():
                # print(entries1)
                indices.append(resume_splitlines.index(entries1)) 
    
    exp_list = ['Experience', 'Work', 'History', 'Career']
    edu_list = ['Educational', 'Education']            
    indices.append(len(resume_splitlines))
    indices = list(set(indices))
    indices.sort()
    
    counter = 0
    for i in indices:
        if i != indices[-1]:
            for entry in exp_list:
                if entry.lower() in resume_splitlines[i].lower():        
                    exp_data = {"Experience": resume_splitlines[indices[counter]+1:indices[counter+1]]}
                    # data.append(d)
                    # print(exp_data)
                    break
            for entry in edu_list:
                if entry.lower() in resume_splitlines[i].lower():
                    edu_data = {"Education": resume_splitlines[indices[counter]+1:indices[counter+1]]}
                    # data.append(d)
                  # print(d)
                    break
            counter+=1
            
        else:
            break
        
    try:
        exp_text = list(exp_data.values())[0]
    except:
        exp_text = resume_splitlines
    # print(exp_text)
    # edu_text = list(edu_data.values())[0]
    # for i in range(len(keys)):
    #     keys[i] = str(keys[i]).replace("[", "")
    #     keys[i] = keys[i].replace("]", "")
    #     keys[i] = keys[i].replace('''"''', "")
    #     keys[i] = keys[i].replace("'","" )
    
    
        
    # with open('data.json', 'w') as outfile:
    #     json.dump(data, outfile)        
        
    
    
    trainDF = pd.read_csv('./sample_sheet.csv')
    trainDF = trainDF.dropna()
    x_train, x_valid, y_train, y_valid = train_test_split(trainDF['Text'], trainDF['Label'])
    
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vect.fit(trainDF['Text'])
    xtrain_tfidf =  tfidf_vect.transform(x_train)
    xvalid_tfidf =  tfidf_vect.transform(x_valid)
    filename = 'finalized_model.sav'
    classifier=naive_bayes.MultinomialNB()
    classifier.fit(xtrain_tfidf, y_train)
    pickle.dump(classifier, open(filename, 'wb'))
    
    # try:
    #     with open('C:\\Users\\rohan\\Desktop\\Fractionable\\Resume\\new\\1487640_656_21_ltresume_19121.txt', encoding = 'utf-8')as f:
    #         resume_data=f.read()
    # except:
    #     with open('C:\\Users\\rohan\\Desktop\\Fractionable\\Resume\\new\\1487640_656_21_ltresume_19121.txt')as f:
    #         resume_data=f.read()        
    
    
    # resume_splitlines=resume_data.splitlines()
    test_input =exp_text
    
    loaded_model = pickle.load(open(filename, 'rb'))
    xtest_tfidf =  tfidf_vect.transform(test_input)
    
    # print(loaded_model.predict_proba(xtest_tfidf))
    # print(loaded_model.predict_proba(tfidf_vect.transform('Perot Systems, Wellesley MA​ - EDI Production Support Analyst')))
    predicted_probabilities = loaded_model.predict_proba(xtest_tfidf)
    
    predictions=[]
    i=0
    for val in predicted_probabilities:
        if val[0] >0.5:
            predictions.append("desc")
        elif val[1]>0.5:
            predictions.append("exp")            
        # else:
        #     predictions.append("Unclear")
        # i+=1
    
    result = []
    temp =[]
    temp_edu = []
    result_edu = []
    for i in range(len(test_input)):
        if predictions[i] == "exp":
            result_edu.append(temp_edu)
            temp_edu = []
            temp.append(test_input[i])
            # if len(test_input[i].split()) < 8:
            #     temp.append(test_input[i].strip())
            # if len(test_input[i].split()) >= 8:
            #     dates = list(datefinder.find_dates(test_input[i]))
            #     location = GeoText(test_input[i])
            #     if dates != [] and location.cities != []:
            #         temp.append(test_input[i].strip())
        # if predictions[i] == 'edu':
        #     temp_edu.append(test_input[i])
        #     result.append(temp)
        #     temp = []
        if i == len(test_input)-1 and predictions[i] == 'exp': 
            result.append(temp)
        if predictions[i] == 'desc':
            # if len(temp)>1:
            result.append(temp)
            result_edu.append(temp_edu)
    
            temp = []
            temp_edu = []
    
    remove_index = []
    for entry in result:
        if len(entry) == 1:
            if len(entry[0].split()) < 4:
                remove_index.append(result.index(entry))
                remove_index.sort()
    for entry in remove_index:
        result[entry] = []
    result = [entry for entry in result if entry != []]    
    
    # for entry in result:
    #     if len(entry)== 1:
    #         loc = GeoText(entry)
    #         if list(datefinder.find_dates(entry)) == [] or loc.cities()== []:
    #             result.remove(entry)
                
    result_edu = [entry for entry in result_edu if entry != []]
    result_dict = {'Abstract': result}

    df = pd.DataFrame(columns = ['Location', 'From', "To", 'Role/Company', 'Total Months'])
    
    # print(result)
    
    
    ind = []
    
    i =0
    k = 0
    To = ''
    From = ''
    abs_result = []
    total_months = 0
    months = 0
    states = ['Alabama','Alaska','American Samoa','Arizona','Arkansas','California','Colorado','Connecticut','Delaware','District of Columbia','Federated States of Micronesia','Florida','Georgia','Guam','Hawaii','Idaho','Illinois','Indiana','Iowa','Kansas','Kentucky','Louisiana','Maine','Marshall Islands','Maryland','Massachusetts','Michigan','Minnesota','Mississippi','Missouri','Montana','Nebraska','Nevada','New Hampshire','New Jersey','New Mexico','New York','North Carolina','North Dakota','Northern Mariana Islands','Ohio','Oklahoma','Oregon','Palau','Pennsylvania','Puerto Rico','Rhode Island','South Carolina','South Dakota','Tennessee','Texas','Utah','Vermont','Virgin Island','Virginia','Washington','West Virginia','Wisconsin','Wyoming', 'AL', 'AK', 'AS', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FM', 'FL', 'GA', 'GU', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MH', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'MP', 'OH', 'OK', 'OR', 'PW', 'PA', 'PR', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VI', 'VA', 'WA', 'WV', 'WI', 'WY' ]  
    month_name = {1:"Jan", 2:'Feb', 3:'Mar',4:'Apr', 5:'May',6:'June',7:'Jul',8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
    month_name_sec = {'01':"Jan", '02':'Feb', '03':'Mar', '04':'Apr', '05':'May', '06':'June', '07':'Jul', '08':'Aug', '09':'Sep'}
    # print(result)
    for entry in result:
        test_time = []
        location = ''
        duration = ''
        total_months = 0
        j =0
        # print(entry)
        for entry1 in entry:
            test_time.append(re.findall(r'(^[Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|Aug(ust)?|Sep(tember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?]+\s+\d{2,4}\s{1,4}(?:to)\s{1,4}[Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|Aug(ust)?|Sep(tember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?]+\s+\d{2,4}$)', entry1.capitalize()))  # Feb 2020 to Feb 2020
            # test_time.append(re.findall(r'(^[Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|Aug(ust)?|Sep(tember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?]+\s+\d{2,4}\s(?:to)\s[Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|Aug(ust)?|Sep(tember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?]+\s+\d{2,4}$)', entry1))  # Feb 2020
            test_time.append(re.findall(r'([Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|Aug(ust)?|Sep(tember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?]+\s+\d{2,4}\s(?:to)\s(?:Present))', entry1.capitalize()))  # Feb 2020
            test_time.append(re.findall(r'([Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|Aug(ust)?|Sep(tember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?]+\s+\d{2,4}\s(?:to)\s(?:present))', entry1.capitalize()))  # Feb 2020
            test_time.append(re.findall(r'([Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|Aug(ust)?|Sep(tember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?]+\s+\d{2,4}\s(?:to)\s(?:Current))', entry1.capitalize()))  # Feb 2020
            test_time.append(re.findall(r'([Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|Aug(ust)?|Sep(tember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?]+\s+\d{2,4}\s(?:to)\s(?:current))', entry1.capitalize()))  # Feb 2020
            
            
            test_time.append(re.findall(r'([Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|Aug(ust)?|Sep(tember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?]+\s+\d{2,4}\s(?:to)\s(?:Till Date))', entry1.capitalize()))  # Feb 2020
            # test_time.append(re.findall(r'([Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|Aug(ust)?|Sep(tember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?]+\s+\d{2,4}\s(?:to)\s(?:till date))', entry1.capitalize()))  # Feb 2020
            # test_time.append(re.findall(r'([Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|Aug(ust)?|Sep(tember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?]+\s+\d{2,4}\s(?:to)\s(?:TILL DATE))', entry1.capitalize()))  # Feb 2020
            # test_time.append(re.findall(r'([Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|Aug(ust)?|Sep(tember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?]+\s+\d{2,4}\s(?:to)\s(?:current))', entry1.capitalize()))  # Feb 2020
            
            
            
            test_time.append(re.findall(r'(^[JAN(UARY)?|FEB(RUARY)?|MAR(CH)?|APR(IL)?|MAY|JUNE(E)?|JUL(Y)?|AUG(UST)?|SEP(TEMBER)?|OCT(OBER)?|NOV(EMBER)?|DEC(EMBER)?]+\s+\d{2,4}\s{1,4}(?:to)\s{1,4}[JAN(UARY)?|FEB(RUARY)?|MAR(CH)?|APR(IL)?|MAY|JUNE(E)?|JUL(Y)?|AUG(UST)?|SEP(TEMBER)?|OCT(OBER)?|NOV(EMBER)?|DEC(EMBER)?]+\s+\d{2,4}$)', entry1.capitalize()))  # FEB 2020 to Feb 2020
            # test_time.append(re.findall(r'(^[Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|Aug(ust)?|Sep(tember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?]+\s+\d{2,4}\s(?:to)\s[Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|Aug(ust)?|Sep(tember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?]+\s+\d{2,4}$)', entry1))  # Feb 2020
            test_time.append(re.findall(r'([JAN(UARY)?|FEB(RUARY)?|MAR(CH)?|APR(IL)?|MAY|JUNE(E)?|JUL(Y)?|AUG(UST)?|SEP(TEMBER)?|OCT(OBER)?|NOV(EMBER)?|DEC(EMBER)?]+\s+\d{2,4}\s(?:to)\s(?:Present))', entry1))  # Feb 2020 to Present
            test_time.append(re.findall(r'([JAN(UARY)?|FEB(RUARY)?|MAR(CH)?|APR(IL)?|MAY|JUNE(E)?|JUL(Y)?|AUG(UST)?|SEP(TEMBER)?|OCT(OBER)?|NOV(EMBER)?|DEC(EMBER)?]+\s+\d{2,4}\s(?:to)\s(?:present))', entry1))  # Feb 2020 to present
            test_time.append(re.findall(r'([JAN(UARY)?|FEB(RUARY)?|MAR(CH)?|APR(IL)?|MAY|JUNE(E)?|JUL(Y)?|AUG(UST)?|SEP(TEMBER)?|OCT(OBER)?|NOV(EMBER)?|DEC(EMBER)?]+\s+\d{2,4}\s(?:to)\s(?:Current))', entry1))  # Feb 2020 to Current
            test_time.append(re.findall(r'([JAN(UARY)?|FEB(RUARY)?|MAR(CH)?|APR(IL)?|MAY|JUNE(E)?|JUL(Y)?|AUG(UST)?|SEP(TEMBER)?|OCT(OBER)?|NOV(EMBER)?|DEC(EMBER)?]+\s+\d{2,4}\s(?:to)\s(?:current))', entry1))  # Feb 2020 to current
            
            test_time.append(re.findall(r'([JAN(UARY)?|FEB(RUARY)?|MAR(CH)?|APR(IL)?|MAY|JUNE(E)?|JUL(Y)?|AUG(UST)?|SEP(TEMBER)?|OCT(OBER)?|NOV(EMBER)?|DEC(EMBER)?]+\s+\d{2,4}\s(?:to)\s(?:Till Date))', entry1))  # Feb 2020 to Till Date
            test_time.append(re.findall(r'([JAN(UARY)?|FEB(RUARY)?|MAR(CH)?|APR(IL)?|MAY|JUNE(E)?|JUL(Y)?|AUG(UST)?|SEP(TEMBER)?|OCT(OBER)?|NOV(EMBER)?|DEC(EMBER)?]+\s+\d{2,4}\s(?:to)\s(?:till date))', entry1))  # Feb 2020 to till date
            test_time.append(re.findall(r'([JAN(UARY)?|FEB(RUARY)?|MAR(CH)?|APR(IL)?|MAY|JUNE(E)?|JUL(Y)?|AUG(UST)?|SEP(TEMBER)?|OCT(OBER)?|NOV(EMBER)?|DEC(EMBER)?]+\s+\d{2,4}\s(?:to)\s(?:Till Now))', entry1))  # Feb 2020 to Till Now
            test_time.append(re.findall(r'([JAN(UARY)?|FEB(RUARY)?|MAR(CH)?|APR(IL)?|MAY|JUNE(E)?|JUL(Y)?|AUG(UST)?|SEP(TEMBER)?|OCT(OBER)?|NOV(EMBER)?|DEC(EMBER)?]+\s+\d{2,4}\s(?:to)\s(?:till now))', entry1))  # Feb 2020 to till now

            
            
            # test_time.append(re.findall(r'[Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|Aug(ust)?|Sep(tember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?].+\s+\d{2,4}', entry1))
            # test_time.append((re.findall(r'(\d{1,2}\s+\d{2,4})', entry1)))                                                                                                                 # 01 2020
            test_time.append((re.findall(r'\d{1,2}/+\d{2,4}\s{1,4}(?:to)\s{1,4}\d{1,2}/+\d{2,4}', entry1))) #01/2014 to 01/2015
            test_time.append((re.findall(r'\d{1,2}/+\d{2,4}\s{1,4}(?:to)\s{1,4}(?:Current)', entry1)))  #01/2014 to Current
            test_time.append((re.findall(r'\d{1,2}/+\d{2,4}\s{1,4}(?:to)\s{1,4}(?:current)', entry1)))  #01/2014 to current
            test_time.append((re.findall(r'\d{1,2}/+\d{2,4}\s{1,4}(?:to)\s{1,4}(?:Present)', entry1)))  #01/2014 to Present
            test_time.append((re.findall(r'\d{1,2}/+\d{2,4}\s{1,4}(?:to)\s{1,4}(?:present)', entry1)))  #01/2014 to present
    
    
            test_time.append((re.findall(r'\d{1,2}/+\d{2,4}\s{1,4}(?:to)\s{1,4}(?:Present)', entry1)))  #
            test_time.append((re.findall(r'\d{1,2}/+\d{2,4}\s{1,4}(?:to)\s{1,4}(?:present)', entry1)))
    
            test_time.append((re.findall(r'\d{1,2}/+\d{1,2}\s{1,4}(?:to)\s{1,4}\d{1,2}/+\d{1,2}', entry1)))
            test_time.append((re.findall(r'\d{1,2}/+\d{1,2}\s{1,4}(?:to)\s{1,4}(?:Present)', entry1)))
            test_time.append((re.findall(r'\d{1,2}/+\d{1,2}\s{1,4}(?:to)\s{1,4}(?:present)', entry1)))
    
    
    
            test_time.append((re.findall(r'\d{4}\s{1,4}(?:to)\s{1,4}\d{4}', entry1)))
    
            # test_time.append((re.findall(r'(\d{2,4}\sto\s+\d{2,4})', entry1)))
            # test_time.append((re.findall(r'(\d{2,4}\s\sto\s\s+\d{2,4})', entry1)))
            # test_time.append((re.findall(r'(\d{2,4}to+\d{2,4})', entry1)))
            test_time = [entry for entry in test_time if entry!= []]
            
            # print(test_time)        
            if test_time != []:
                duration = test_time
                for date in duration:
                    entry1 = entry1.replace(date[0], '')
                # ind.append(entry.index(entry1))
                # test_time = []
           
            
                        
            temp = GeoText(entry1)
            try:
                if temp.cities != []:
                    location = temp.cities
                    if len(entry1.split()) < 4:
                        ind.append(entry.index(entry1))
            except:
                pass
            
            if location == '':
                for ent in entry1.split():
                    for state in states:
                        if state.lower() == ent.lower():
                            location = state
                            # if len(entry1.split()) < 4:
                            #     ind.append(entry.index(entry1))
                
            ind = list(set(ind))
            # print(entry)
            # print(duration)
            for index in ind:
                entry[index] = ''
            entry = [en for en in entry if en != '']
            # entry = ", ".join(entry)
            
                
            role = str(entry)
            if type(location) == list:
                location = " ".join(location)
            # for dates in test_time:
            #     for dates1 in dates:
            #         role = role.replace(dates1, '')
            try:
                if duration == [''] or duration == '' or duration == []:
                    # print(entry)
                    entry_str = str(entry)
                    a = list(datefinder.find_dates(entry_str))
                    # print(a)
                    if len(a) == 2:
                        # print(True)
                        from_date = a[0]
                        from_month = from_date.month
                        if from_date.year > 1980 and from_date.year < datetime.datetime.now().year:
                            from_year = from_date.year
                        
                        from_month = month_name.get(from_month)
                        
                        
                        From = str(from_month)+' '+str(from_year)
                        # print(From)
                        to_date = a[1]
                        # if (to_date == "Current" or to_date == "Present" or to_date == "current" or to_date== "present"):
                        #     to_date = datetime.datetime.now()
                        to_month = to_date.month
                        
                        if to_date.year > 1980 and to_date.year < datetime.datetime.now().year:
                            to_year = to_date.year
                        
                        to_month = month_name.get(to_month)
                        
                        
                        To = str(to_month)+' '+str(to_year)
                        # print(To)
                        total_days = (to_date - from_date).days
                        total_months = abs(int(total_days/30))
                        From = From.capitalize()
                        To = To.capitalize()
                        role = role.replace(To, '')
                        role = role.replace(From, '')
                        role = role.replace(' to ', ' ')
                        role = role.replace('till', '')
                        
                        
                        # print(role)
                        # print(from_date, to_date)
                        
                        # total_year = total_months//12
                        # month = total_months%12
                        # total_duration = str(total_year)+'.'+str(month)+" years"
                else:
                # if duration != [''] or duration != '' or duration != []:
                    try:
                        # print('Second case')
                        duration = str(duration)
                        duration = duration.replace('[','')
                        duration = duration.replace(']', '')
                        duration = duration.replace("'", '')
                        duration = duration.split(' to ')
                        From = duration[0]
                        To = duration[-1]

                        if "/" in From:
                            From_split = From.split('/')
                            From = month_name.get(int(From_split[0]))
                            if int(From_split[1]) > 1980 and int(From_split[1]< datetime.datetime.now().year):
                                From = From + ' ' + From_split[1]
                            # if From:
                            #     pass
                            # else:
                            #     From = month_name_sec.get(From[0])
                        if '/' in To:
                            To_split = To.split('/')
                            To = month_name.get(int(To_split[0]))
                            
                            if int(To_split[1]) > 1980 and int(To_split[1]< datetime.datetime.now().year):
                                 To = To + ' ' + To_split[1]
                            # if To:
                            #     pass
                            # else:
                            #     To = month_name_sec.get(To[0])
                                
                        # print(To)
                        from_duration = list(datefinder.find_dates(From))[0]
                        # if 'present' or 'current' or 'Present' or 'Current' in To:
                        #     # print("Hello")
                        #     To = datetime.datetime.now()
                        #     total_days = (To - from_duration).days
                        #     total_months = abs(int(total_days/30))
                        #     To = str(month_name.get(datetime.datetime.now().month)) +" "+str(datetime.datetime.now().year)
                        # else:
                        to_duration = list(datefinder.find_dates(To))[0]
                        total_days = (to_duration - from_duration).days
                        total_months = abs(int(total_days/30))
                        
                        
                        From = From.capitalize()
                        To = To.capitalize()
                        role = role.replace(To, '')
                        role = role.replace(From, '')
                        role = role.replace(' to ', ' ')
                        role = role.lower().replace('present', '')
                       
                        # if To == "Current" or To == "Present" or To == "current" or To == "present":
                        
                        # total_year = total_months//12
                        # month = total_months%12
                        # total_duration = str(total_year)+'.'+str(month)+" years"
                    except:
                        pass
            except:
                pass
            for time in test_time:
                for time1 in time:
                    role = role.replace(time1, '')
                    # role = role.replace(To, '')
                    role = role.replace(' to ', ' ')
                    role = role.lstrip()
            # print(role)
            # temp_df = pd.DataFrame({"Location": location, "From": From, "To": To, "Role/Company": role, 'Total Months': total_months}, index = [k])
            # df = df.append(temp_df)
        if total_months == 0:
            try:
                total_days = (list(datefinder.find_dates(To))[0] - list(datefinder.find_dates(From))[0]).days
                total_months = total_days//30
            except:
                pass
        # role = ", ".join(role)
        # role = (",".join([str(role[i]) for i in range(len(role))]))
        role = role.replace(location, '')
        role1 = ''
        for entry in role:
            role1 = role1 + entry
        role1 = role1.replace("[",'')
        role1 = role1.replace(']', '')
        role1 = role1.replace("'", "")
        role1 =  re.sub(r'([Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|Aug(ust)?|Sep(tember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?]\s{1,4}\d{2,4})', '', role1)
        
        # print(role1)
        temp_list = re.findall(r'([Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|Aug(ust)?|Sep(tember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?]+\s+\d{2,4}\s{1,4}(?:to)\s{1,4}[Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|Aug(ust)?|Sep(tember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?]+\s+\d{2,4})', role1)
        # print(temp_list)
        try:
            role1 = role1.replace(temp_list[0], '')
        except:
            pass
                
        
        abs_result.append([location,From, To, role1, abs(total_months)])
        # print(abs_result)
        ind = []
        
        k+=1
        months = months + total_months
            # df = df.append(temp_df)
            # ind = []
        
    #     k+=1
    #     # print(To, From)
    # df['total_exp'] = df['Total Months'].sum()
        
            
    # df       
    # with open('exp.json', 'w') as file:
    #     json.dump(result_dict, file)
    



    ################################################ LOCATION #####################################################################################################
    location = ''
    #find the current location based on what the candidate has mentioned in their resume
    states = ['Alabama','Alaska','American Samoa','Arizona','Arkansas','California','Colorado','Connecticut','Delaware','District of Columbia','Federated States of Micronesia','Florida','Georgia','Guam','Hawaii','Idaho','Illinois','Indiana','Iowa','Kansas','Kentucky','Louisiana','Maine','Marshall Islands','Maryland','Massachusetts','Michigan','Minnesota','Mississippi','Missouri','Montana','Nebraska','Nevada','New Hampshire','New Jersey','New Mexico','New York','North Carolina','North Dakota','Northern Mariana Islands','Ohio','Oklahoma','Oregon','Palau','Pennsylvania','Puerto Rico','Rhode Island','South Carolina','South Dakota','Tennessee','Texas','Utah','Vermont','Virgin Island','Virginia','Washington','West Virginia','Wisconsin','Wyoming', 'AL', 'AK', 'AS', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FM', 'FL', 'GA', 'GU', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MH', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'MP', 'OH', 'OK', 'OR', 'PW', 'PA', 'PR', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VI', 'VA', 'WA', 'WV', 'WI', 'WY' ]
    location_state = '-NA-'
    location_city = '-NA-'
    zipcode = 0
    splitlines1 = text.splitlines()
    splitlines1 = [entry for entry in splitlines1 if entry!= ""]
#    print(splitlines1)
    
    
    cities = pd.read_csv('cities.csv')
    cities = cities.fillna('')
    cities = cities.to_dict('list')
    
    starting_lines = "".join(resume_splitlines[0:6])
    # print(starting_lines)
    
    cities_name = list(cities.values())
    i = 0
    for entry in cities_name:
        cities_name[i] = [city for city in entry if city != '']
        i= i+1
    cities_name = (list(itertools.chain.from_iterable(cities_name)))
    
    
    def word2vec(word):
    
        # count the characters in word
        cw = Counter(word)
        # precomputes a set of the different characters
        sw = set(cw)
        # precomputes the "length" of the word vector
        lw = sqrt(sum(c*c for c in cw.values()))
    
        # return a tuple
        return cw, sw, lw
    
    def cosdis(v1, v2):
        # which characters are common to the two words?
        common = v1[1].intersection(v2[1])
        # by definition of cosine distance we have
        return sum(v1[0][ch]*v2[0][ch] for ch in common)/v1[2]/v2[2]
    def get_key(val): 
        for key, value in cities.items():
            for city in value:
                if val == city: 
                    return key



    if resume_source == 'DICE':
        for i in range(len(splitlines1)):
            if 'contact information' in splitlines1[i].lower():
                search = splitlines1[i-1]
                search = search.replace("."," ")
                search = search.replace(","," ")
#                print(splitlines1[i-1])
                places = GeoText(search)
                try:
                    location_city = places.cities[0]
                except:
                    location_city = '-NA-'

                try:
                    zipcode = re.findall(r"\b\d{5}\b", "".join(splitlines1[0:7]))[0]
                except:
                    zipcode = 0
                try:
                    for entry in search.split():
                        if entry in states:
                            location_state = entry
                            break
                except:
                    location_state = '-NA-'

            else:
                search = splitlines1[0:6]
                search = "".join(search)
                search = search.replace("."," ")
                search = search.replace(","," ")

            #    print(search)
                places = GeoText(search)
                location_city = '-NA-'
                zipcode = 0
                try:
                    location_city = places.cities[0]

                except:
                    location_city = '-NA-'
                try:
                    zipcode = re.findall(r"\b\d{5}\b", search)[0]
                except:
                    zipcode = 0
                try:
                    for entry in search.split():
                        if entry in states:
                            location_state = entry
                            break
                except:
                    location_state = '-NA-'
    

    
    if not location :                
        for city in cities_name:
            for word in starting_lines.split():
                city_vec = word2vec(city)
                word_vec = word2vec(word)
                if cosdis(city_vec,word_vec) > 0.95:
                    location_city = city
                    location_state = get_key(location_city)
                    # print(word, location_city)
                    break
    if location_state =='-NA-' and location_city == '-NA-':
        starting_lines = str(abs_result[0][0])
        for city in cities_name:
            for word in starting_lines.split():
                city_vec = word2vec(city)
                word_vec = word2vec(word)
                if cosdis(city_vec,word_vec) > 0.95:
                    location_city = city
                    location_state = get_key(location_city)
                    # print(word, location_city)
                    break
        
    location = location_city 
    
            


    ############################################################## LATEST PROJECT INFO ###########################################################################
    key_words = ['till date', 'present', 'till now', "current"]

    splitlines = resume_text.splitlines()

    splitlines = [entry for entry in splitlines if entry != '']
    latest_project = ' -NA- '
    try:
        for i in range(len(splitlines)):
            for words in key_words:
                if words in splitlines[i].lower():
            #            print(splitlines[i-1])
                    latest_project = " ".join(splitlines[i-1:i+1])
        #            print(splitlines[i+1])     \
    except:
        latest_project = " -NA- "

    ################################## Skills ###############################################################################
    
    skills = []

    resume_text = resume_text.replace(',', '')
    for entry in resume_text.split():
        if entry.lower() in skill_roster:
            skills.append(entry)
    skills = [entry.lower() for entry in skills]
    skills = list(set(skills))
    skills = [entry.capitalize() for entry in skills]
    
    
    ################################## VISA STATUS ###############################################################################
    visa = ' Unknown '

    for lines1 in resume_splitlines:
        if "h1" in lines1.lower():
            visa = 'H1-B'
            break
        elif 'us citizenship' in lines1.lower():
            visa = 'US CITIZENSHIP'
            break




    ################################## TOTAL EXP ###############################################################################
    tot_exp = '-NA-'
    for line in result:
        # print(line)
        tot_exp =  re.findall(r'[0-9]\s{1,3}(?:years)', str(line))
        if tot_exp != []:
            tot_exp = str(tot_exp)
            tot_exp = tot_exp.replace('[', '')
            tot_exp = tot_exp.replace(']', '')
            tot_exp = tot_exp.replace("'", '')
            tot_exp = tot_exp.replace('years', '')
            break
            
        

    ############################################ DICTIONARY ###############################################################################
    resdict =   {"name":name_str ,
             "email":(email),
             "phone":phoneno,
              # "certifications" : certification_list,
              "location" : location,
#              "state": location_state,
              "postal": zipcode,
              "abstract": abs_result,
              "visa": visa,
              "skills": skills,
              "total_exp" : str(tot_exp),
              "resume_source": resume_source}


    
    return(resdict)

#filename = 'Srinivas-Donikena.txt'
#resume_path =  'C:/Users/Rohan joshi/Desktop/Resume Parsing Score/Resumes' + filename
#try:
#    resume_text = open(resume_path).read()
#except:
#    resume_text = open(resume_path, encoding = 'utf-8').read()

#skill_roster = pd.read_csv('skills.csv')
#skill_roster = list(skill_roster['Skills'])
#skill_roster = [entry.lower() for entry in skill_roster]
#info = extract_info(skill_roster, resume_text)
#print("{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in info.items()) + "}")

