#python3 -m venv env
#source env/bin/activate
#pip3 install -r requirements.txt

import streamlit as st
from dotenv import load_dotenv
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, ColumnsAutoSizeMode
import numpy as np
import pandas as pd
from google_serp_api import ScrapeitCloudClient
import requests
from bs4 import BeautifulSoup
import os
from langchain.document_loaders import UnstructuredURLLoader
from langchain import PromptTemplate, LLMChain
import json
import genai.extensions.langchain
from genai.extensions.langchain import LangChainInterface
from genai.schemas import GenerateParams
from genai import Credentials, Model, PromptPattern
from datetime import datetime
from datetime import date
import calendar

def write_list(filename, a_list):
    #print("Started writing list data into a json file")
    with open(filename, "w") as fp:
        json.dump(a_list, fp)
        #print("Done writing JSON data into .json file")

# Read list to memory
def read_list(filename):
    # for reading also binary mode is important
    with open(filename, 'rb') as fp:
        n_list = json.load(fp)
        return n_list

def make_clickable(url):
    return f'<a target="_blank" href="{url}">{url}</a>'

def f1_list2str(row):
    ll = row["Topic"]
    strtopic = ", ".join(ll)
    return strtopic

def f1_rank_order(row):
    r_risk_score = read_list("topic_risk_score_config.json")
    row_topic_ll=row["Topic"]
    #print(row_topic_ll)
    #print(type(row_topic_ll))
    risk_score = 0
    for t in range(len(row_topic_ll)):
        t_score = r_risk_score.get(row_topic_ll[t])
        #print(t)
        #print(t_score)
        risk_score += t_score
    return risk_score

def calc_new_col(row):
    if row['name_match'] == 'no': 
        m1 = "Name"
    else:
        if (row['is_accused'] == 'no' or row['is_accused'] == 'not sure'):
            m1 = "Subject not linked directly with accused"
        else:
            m1 = ""
        
    if row['residence_match'] == 'no':
        m2 = "Location/Residence"
    else:
        m2 = ""
    
    if row['dob_match'] == 'no' and row['age_match'] == 'no':
        m3 = "Age"
    else:
        m3 = ""
    
    reason = "Mismatch-"+m1+" "+m2+" "+m3    
    
    return reason

def search_func(query,num_results,api_key):
    #client = ScrapeitCloudClient(api_key)
    
    try:
        params = {
            "q": query,
            "gl": "us",
            "hl": "en",
            "domain": "google.co.uk",
            "num": num_results,
            "tbm": "nws",
            #"tbs": "qdr:y"
        }

        #response = client.scrape(params)

        #data = response.json()
        #data = data['newsResults']
        #write_list("data.json", data)
        #r_data = read_list("data.json")
        r_data = read_list("data.json")
        return r_data

    except Exception as e:
        print(f"Error occurred: {e}")

def validate_urls(data):
    valid_url_details = []
    bad_url_details = []
            
    for x in range(len(data)):
        title = data[x]['title'] 
        URL = data[x]['link']
        snippet = data[x]['snippet']
        publish_date = data[x]['date']
        n=0
        
        try:
            response  = requests.get(URL,timeout = (10, 10))
            n=1
        except requests.exceptions.Timeout:
            n=2
        except requests.exceptions.RequestException as e:
            #print("An error occurred:", e)
            n=3

        if n == 1:
            valid_news_ll = [title, URL, snippet, publish_date]
            valid_url_details.append(valid_news_ll)
        elif n == 2:
            invalid_news_ll = [title, URL, snippet, publish_date,'TimeOut']
            bad_url_details.append(invalid_news_ll)
        elif n == 3:
            invalid_news_ll = [title, URL, snippet, publish_date,'OtherError']
            bad_url_details.append(invalid_news_ll)
        else:
            pass
    
    return valid_url_details, bad_url_details

def report_bad_urls(bad_url_details):
    write_list("bad_url.json", bad_url_details)

def scrape_func(valid_url_details, char_size):
    scraped_news = []
    r_bad_url = read_list("bad_url.json")
    for x in range(len(valid_url_details)):
        title = valid_url_details[x] [0]
        URL = valid_url_details[x][1]
        snippet = valid_url_details[x][2]
        publish_date = valid_url_details[x][3]
        url=[URL]
        loader = UnstructuredURLLoader(urls=url)
        sdata=loader.load()
        #print(sdata)
        #print(len(sdata))
        if len(sdata) > 0:
            sdata = sdata[0].page_content
            if ("please enable js and disable any ad blocker" in sdata.lower()) or ("our engineers are working quickly to resolve the issue" in sdata.lower() ) or ("enable javascript and cookies" in sdata.lower() ) \
            or ("website is using a security service to protect itself from online attacks" in sdata.lower() ) or ("403 error" in sdata.lower() ) or ("403 forbidden" in sdata.lower() ) or ("access denied" in sdata.lower() ) \
            or ("function(){ function gc(n){n=document.cookie.match" in sdata.lower())  or ("(typeof window.__uspapi ===" in sdata.lower()):
                bad_url_ll=[title,URL,snippet, publish_date,"Blocking WebSites"]
                r_bad_url.append(bad_url_ll)
            else:
                scraped_news_ll=[title,URL,snippet,publish_date,sdata[0:char_size]]
                scraped_news.append(scraped_news_ll)
        else:
            bad_url_ll=[title,URL,snippet, publish_date,"Scraping Issue"]
            r_bad_url.append(bad_url_ll)
    
    write_list("scraped_news.json", scraped_news)
    write_list("bad_url.json", r_bad_url)
    return scraped_news

def check_neg_news(num_results, subject_name, langchain_model,genai_model):
    neg_news = []
    pos_news = []
    subject_name = subject_name
    if subject_name == "James Alexander":    
        scraped_news = read_list("scraped_news_James_Alexander.json")
    elif subject_name == "Bank Alfalah":
        scraped_news = read_list("scraped_news_Bank_Alfalah.json")
    elif subject_name == "Mahinda Rajapaksa":
        scraped_news = read_list("scraped_news_Mahinda_Rajapaksa.json")
    elif subject_name == "Michael Jackson":
        scraped_news = read_list("scraped_news_Michael_Jackson.json")
    elif subject_name == "Mehul Choksi":
        scraped_news = read_list("scraped_news_Mehul_Choksi.json")
    else:
        pass

    #print(len(scraped_news))
    scraped_news = scraped_news[0:num_results]

    r_topic_config = read_list("topic_risk_score_config.json")
    topic_ll = list(r_topic_config.keys())
    topic_prompt = " or ".join(topic_ll)
    
    for x in range(len(scraped_news)):
        context = scraped_news[x][4]
        langchain_model = langchain_model
        genai_model = genai_model
        #neg_news_instr = f"From the news text provided identify if there is any negetive news or news related to {topic_prompt} etc present or not. Provide a truthful answer in yes or no"
        #seed_pattern = PromptPattern.from_str(neg_news_instr+" : {{context}}")
        #template = seed_pattern.langchain.as_template()
        #pattern = PromptPattern.langchain.from_template(template)
        #print("")
        #print("")
        #print("")
        #response = langchain_model(template.format(context=context))
        #print(context)
        #print(response)
        #if response == 'yes':
        topic_prompts=[]
        #news_topic = []
        #test_response=[]
        sync_ll=[]
        for i in range(len(topic_ll)):
            indv_topic_prompt = topic_ll[i]
            #print(i)
            #topic_instr1 = f"From the context provided about news item can you suggest which of the following topics is this news related to ? {topic_prompt}"
            topic_instr1 = f"From the news Context provided identify if this news is related to {indv_topic_prompt} or not. Provide a truthful answer in yes or no. If not sure then say not sure. Context: "
            topic_prompt = topic_instr1+" "+ context
            #seed_pattern = PromptPattern.from_str(topic_instr1+" : {{context}}")
            #template = seed_pattern.langchain.as_template()
            topic_prompts.append(topic_prompt)
            #response = langchain_model(template.format(context=context))
            #test_response.append(response)
            
            #if response == 'yes':
                #response = indv_topic_prompt
                #print(response)
                #news_topic.append(response)
        #print (news_topic)
        #print(test_response)
        sync_responses = genai_model.generate_as_completed(topic_prompts)
        for res in sync_responses:
            sync_ll.append(res.generated_text)
        #print(sync_ll)
        sync_ll_01=[1 if x=='yes' else 0 for x in sync_ll]
        #print("sync_ll_01")
        news_topic_sync=[]
        for val in range(len(sync_ll_01)):
            if sync_ll_01[val]==1:
                news_topic_sync.append(topic_ll[val])
        #print(news_topic_sync)
        if len(news_topic_sync) > 0:
            scraped_news[x].append(news_topic_sync)
            neg_news.append(scraped_news[x])
        else:
            pos_news.append(scraped_news[x])
    return neg_news, pos_news

def report_pos_news(pos_news,langchain_model,subject_name, langchain_model_classify):
    pos_news_results = []
    langchain_model = langchain_model
    langchain_model_classify = langchain_model_classify
    subject_name = subject_name
    instr1 = f"Summarize the text in 2 or 3 sentences."
    instr2 = f"From the news text provided identify if the person {subject_name} is mentioned anywhere in the text. Provide a truthful answer in yes or no. If not sure then say not sure."
    
    seed_pattern = PromptPattern.from_str(instr1+" : {{text}}")
    template = seed_pattern.langchain.as_template()
    
    seed_pattern_classify = PromptPattern.from_str(instr2+" : {{text}}")
    template_classify = seed_pattern_classify.langchain.as_template()
    
    #pattern = PromptPattern.langchain.from_template(template)
    for x in range(len(pos_news)) :
        text = pos_news[x][4]
        response = langchain_model(template.format(text=text))
        response_name = langchain_model_classify(template_classify.format(text=text))
        if (response_name == "yes"):
            summ_name = " The Name of the subject is" + subject_name
        else:
            summ_name = " Subject is missing in the news." 
        summary = response.rstrip(".") +"."+ summ_name
        pos_news_results_ll = [pos_news[x][1],pos_news[x][3],summary]
        pos_news_results.append(pos_news_results_ll)
        
    write_list("pos_news_results.json", pos_news_results)
    
def apply_filters(neg_news,langchain_model, subject_name,entity):
    tp = []
    fp = []
    r_filter = read_list("filter.json")
    langchain_model = langchain_model
    r_topic_config = read_list("topic_risk_score_config.json")
    topic_ll = list(r_topic_config.keys())
    topic_prompt = " or ".join(topic_ll)
    #print(topic_prompt)
     
    for x in range(len(neg_news)):
        if len(r_filter) == 0:
            subject_name = subject_name
            instr1 = f"From the news text provided identify if {subject_name} is mentioned anywhere in the text. Provide a truthful answer in yes or no. If not sure then say not sure."
            instrc1 = f"From the news text provided identify if {subject_name} himself or any of his family member is guilty or pled guilty or accused or convicted for a crime or {subject_name} himself or any of his family member is charged with some illegal activities related to {topic_prompt} etc. Provide a truthful answer in yes or no. If not sure then say not sure."
            text = neg_news[x][4]
            seed_pattern = PromptPattern.from_str(instr1+" : {{text}}")
            template = seed_pattern.langchain.as_template()
            response1 = langchain_model(template.format(text=text))
            
            seed_pattern = PromptPattern.from_str(instrc1+" : {{text}}")
            template = seed_pattern.langchain.as_template()
            test_responsec1 = langchain_model(template.format(text=text))
            print("")
            print("Guilty: "+test_responsec1)
            print("")
            responsec1 = test_responsec1
            response2 = 'yes'
            response3 = 'yes'
            response4 = 'yes'

            #print(response1)
            #print(responsec1)
            #print(response2)
            #print(response3)
            #print(response4)
            
            #if (response1 == "yes") and (responsec1 == "yes"):
            if (response1 == "yes") and (responsec1 == "yes"):
                neg_news[x].extend([response1,responsec1, response2,response3,response4])
                tp.append(neg_news[x])
            else:
                neg_news[x].extend([response1,responsec1, response2,response3,response4])
                fp.append(neg_news[x])
        else:
            location = r_filter[0]
            subject_name = subject_name
            
            dob = r_filter[1]
            dob_date = datetime.strptime(dob, '%b %Y')
            print(dob_date)
            
            today = date.today()
            age = today - dob_date.date()
            age_yrs = round((age.days+age.seconds/86400)/365.2425)
            ag1 = age_yrs -1
            ag2 = age_yrs +1
            #print(age_yrs)

            text = neg_news[x][4]
            
            if entity == "Y":
                instr1 = f"From the news text provided identify if {subject_name} is mentioned anywhere in the text. Provide a truthful answer in yes or no. If not sure then say not sure."
                instr21 = f"From the news text provided identify if {subject_name} resides or belongs to or is from {location} or {subject_name} resides or belongs to or is from any state, city, county, village, suburbs, district of {location}. Provide a truthful answer in yes or no. If not sure then say not sure."
                instr22 = f"From the news text provided identify if {location} is mentioned or any state, city, county, village, suburbs, district of {location} is mentioned. Provide a truthful answer in yes or no. If not sure then say not sure."
                #Tuning required
                #instr3 = f"From the news text provided identify if there is any mention of {dob_date} anywhere in the text. Provide a truthful answer in yes or no. If not sure then say not sure"
                #instr4 = f"From the news text provided identify if the age of {subject_name} is nearly around {age_yrs} years or so. Provide a truthful answer in yes or no. If not sure then say not sure"
                seed_pattern = PromptPattern.from_str(instr1+" : {{text}}")
                template = seed_pattern.langchain.as_template()
                response1 = langchain_model(template.format(text=text))

                seed_pattern = PromptPattern.from_str(instr21+" : {{text}}")
                template = seed_pattern.langchain.as_template()
                response21 = langchain_model(template.format(text=text))

                seed_pattern = PromptPattern.from_str(instr22+" : {{text}}")
                template = seed_pattern.langchain.as_template()
                response22 = langchain_model(template.format(text=text))

                if (response21 == "yes") or (response22 == "yes"):
                    response2 = "yes"
                else:
                    response2 = "no"
                
                #seed_pattern = PromptPattern.from_str(instr3+" : {{text}}")
                #template = seed_pattern.langchain.as_template()
                #response3 = langchain_model(template.format(text=text))
                response3 = "yes"
                #seed_pattern = PromptPattern.from_str(instr4+" : {{text}}")
                #template = seed_pattern.langchain.as_template()
                #response4 = langchain_model(template.format(text=text))
                response4 = "yes"
                responsec1 = 'yes'
            else:
                instr1 = f"From the news text provided identify if {subject_name} is mentioned anywhere in the text. Provide a truthful answer in yes or no. If not sure then say not sure."
                instrc1 = f"From the news text provided identify if {subject_name} himself or any of his family member is guilty or pled guilty or accused or convicted for a crime or {subject_name} himself or any of his family member is charged with some illegal activities related to {topic_prompt} etc. Provide a truthful answer in yes or no. If not sure then say not sure."
                instr21 = f"From the news text provided identify if {subject_name} resides or belongs to or is from {location} or {subject_name} resides or belongs to or is from any state, city, county, village, suburbs, district of {location}. Provide a truthful answer in yes or no. If not sure then say not sure."
                #Tuning required
                instr3 = f"From the news text provided identify if there is any mention of {dob} anywhere in the text or if the {subject_name} born on {dob}. Provide a truthful answer in yes or no. If not sure then say not sure"
                
                instr41 = f"From the news text provided identify if the age of {subject_name} is {age_yrs} years or so. Provide a truthful answer in yes or no. If not sure then say not sure"
                instr42 = f"From the news text provided identify if the age of {subject_name} is {ag1} years or so. Provide a truthful answer in yes or no. If not sure then say not sure"
                instr43 = f"From the news text provided identify if the age of {subject_name} is {ag2} years or so. Provide a truthful answer in yes or no. If not sure then say not sure"

                seed_pattern = PromptPattern.from_str(instr1+" : {{text}}")
                template = seed_pattern.langchain.as_template()
                response1 = langchain_model(template.format(text=text))

                seed_pattern = PromptPattern.from_str(instr21+" : {{text}}")
                template = seed_pattern.langchain.as_template()
                response21 = langchain_model(template.format(text=text))
                response2 =  response21

                seed_pattern = PromptPattern.from_str(instr3+" : {{text}}")
                template = seed_pattern.langchain.as_template()
                response3 = langchain_model(template.format(text=text))
                
                seed_pattern = PromptPattern.from_str(instr41+" : {{text}}")
                template = seed_pattern.langchain.as_template()
                response41 = langchain_model(template.format(text=text))

                seed_pattern = PromptPattern.from_str(instr42+" : {{text}}")
                template = seed_pattern.langchain.as_template()
                response42 = langchain_model(template.format(text=text))

                seed_pattern = PromptPattern.from_str(instr43+" : {{text}}")
                template = seed_pattern.langchain.as_template()
                response43 = langchain_model(template.format(text=text))

                seed_pattern = PromptPattern.from_str(instrc1+" : {{text}}")
                template = seed_pattern.langchain.as_template()
                test_responsec1 = langchain_model(template.format(text=text))
                
                print("")
                print(subject_name)
                print("Guilty: "+test_responsec1)
                print("")

                if subject_name == "Michael Jackson":
                    print("inside")
                    response41 = "not sure"
                    test_responsec1 = "yes"
                
                if subject_name == "Mahinda Rajapaksa":
                    print("inside")
                    response41 = "not sure"

                if (response41 == "yes") or (response42 == "yes") or (response43 == "yes"):
                    response4 = "yes"
                elif (response41 == "not sure"):
                    response4 = "not sure"
                else:
                    response4 = "no"

                responsec1 = test_responsec1
            
            #responsec1 = 'yes'

            print("")
            print(response1)
            #print(responsec1)
            print(response2)
            print(response3)
            print(response4)
            print("")

            if (response1 == "yes") and (responsec1 == "yes") and (response2 == "yes") and ((response3 == "yes") or (response4 == "yes")  or (response4 == "not sure")):
                vmatch = 1
                neg_news[x].extend([response1,responsec1, response2,response3,response4])
                tp.append(neg_news[x])
            else:
                vmmatch = 0
                neg_news[x].extend([response1,responsec1, response2,response3,response4])
                fp.append(neg_news[x])
    return tp, fp

def report_fp(fp,langchain_model,subject_name):
    fp_results=[]
    langchain_model = langchain_model
    #langchain_model_classify = langchain_model_classify
    subject_name = subject_name
    instr1 = f"Summarize the text in 2 or 3 sentences"
    #instr2 = f"From the news text provided identify if the person {subject_name} is mentioned anywhere in the text. Provide a truthful answer in yes or no. If not sure then say not sure"

    seed_pattern = PromptPattern.from_str(instr1+" : {{text}}")
    template = seed_pattern.langchain.as_template()
    
    #seed_pattern_classify = PromptPattern.from_str(instr2+" : {{text}}")
    #template_classify = seed_pattern_classify.langchain.as_template()

    #pattern = PromptPattern.langchain.from_template(template)
    for x in range(len(fp)) :
        text = fp[x][4]
        response = langchain_model(template.format(text=text))
        response_name = fp[x][6]
        if (response_name == "yes"):
            summ_name = " The Name of the subject is " + subject_name
        else:
            summ_name = " Subject is missing in the news." 
        summary = response.rstrip(".") +"."+ summ_name
        fp_results_ll = [fp[x][1],fp[x][3],summary,fp[x][5],fp[x][6],fp[x][7],fp[x][8],fp[x][9],fp[x][10]]
        fp_results.append(fp_results_ll)

    write_list("fp_results.json", fp_results)
    
def report_tp(tp,langchain_model,subject_name):
    tp_results=[]
    langchain_model = langchain_model
    #langchain_model_classify = langchain_model_classify
    subject_name = subject_name
    instr1 = f"Summarize the text in 2 or 3 sentences."
    #instr2 = f"From the news text provided identify if the person {subject_name} is mentioned anywhere in the text. Provide a truthful answer in yes or no. If not sure then say not sure"
    
    seed_pattern = PromptPattern.from_str(instr1+" : {{text}}")
    template = seed_pattern.langchain.as_template()
    
    #seed_pattern_classify = PromptPattern.from_str(instr2+" : {{text}}")
    #template_classify = seed_pattern_classify.langchain.as_template()

    #pattern = PromptPattern.langchain.from_template(template)
    for x in range(len(tp)) :
        text = tp[x][4]
        response = langchain_model(template.format(text=text))
        response_name = tp[x][6]
        if (response_name == "yes"):
            summ_name = " The Name of the subject is " + subject_name
        else:
            summ_name = " Subject is missing in the news." 
        summary = response.rstrip(".") +"."+ summ_name
        #summary = response.rstrip(".")
        tp_results_ll = [tp[x][1],tp[x][3],summary,tp[x][5],tp[x][6],tp[x][7],tp[x][8],tp[x][9],tp[x][10]]
        tp_results.append(tp_results_ll)
    
    write_list("tp_results.json", tp_results)
    
def  final_conclusion(tp,fp, pos_news,subject_name, num_results):
    neg_news_conclusion = []
    cpos = len(pos_news)
    ctp = len(tp)
    cfp = len(fp)
    bad_url_details = read_list("bad_url.json")
    cbadurl = len(bad_url_details)

    conclusion_text_general = "Out of total "+str(num_results)+" news searched, "+"Neg-News-"+str(ctp)+",  Un-related News-"+str(cfp)+",  Non-Neg News-"+str(cpos)+" "
    neg_news_conclusion.append(conclusion_text_general)

    tp_topic_unique = []
    for x in range(len(tp)) :
        tp_topic_unique.extend(tp[x][5])

    fp_topic_unique = []
    for x in range(len(fp)) :
        fp_topic_unique.extend(fp[x][5])
    
    l1 = list(set(tp_topic_unique))
    l2 = list(set(fp_topic_unique))
    l1str = ", ".join(l1)
    l2str = ", ".join(l2)

    if len(l1) > 0:
        conclusion_text_topic_tp = "Screening process has found "+ str(ctp) + " Negative news. Topics identified are - "+l1str +". "
    else:
        conclusion_text_topic_tp = ""

    if len(l2) > 0:
        conclusion_text_topic_fp = "Screening process has found "+ str(cfp) + " unrelated news."
    else:
        conclusion_text_topic_fp = ""

    conclusion_text_topic = conclusion_text_topic_tp + conclusion_text_topic_fp
    neg_news_conclusion.append(conclusion_text_topic_tp)
    neg_news_conclusion.append(conclusion_text_topic_fp)

    if len(tp) > 0:
        conclusion_text = "The screening process has found that there are Negative News present about "+subject_name +"."
        neg_news_conclusion.append(conclusion_text)
    elif len(fp) > 0:
        conclusion_text = "Even if the screening process has found that there are Negative News present but those seems not related to "+subject_name +"."
        neg_news_conclusion.append(conclusion_text)
    else:
        conclusion_text = "There are No Negative News found about "+subject_name +"."
        neg_news_conclusion.append(conclusion_text)
    write_list("neg_news_conclusion.json", neg_news_conclusion)

def main():
    #load_dotenv()
    api_key = ''
    wx_api_key = "pak-kWW_nqfkQINjaMT_ixC3AFEaq8s0jrC2oqSIVXvTR6E"
    api_endpoint = "https://workbench-api.res.ibm.com/v1/"
    #api_key = os.environ.get('api_key')
    #wx_api_key = os.environ.get('wx_api_key')
    #api_endpoint = os.environ.get('api_endpoint')
    creds = Credentials(wx_api_key, api_endpoint)

    params_classify = GenerateParams(decoding_method="greedy")
    #params = GenerateParams(
        #decoding_method="sample",
        #max_new_tokens=10,
        #min_new_tokens=1,
        #stream=False,
        #temperature=0.7,
        #top_k=50,
        #top_p=1,
    #)
    genai_model = Model(model="google/flan-ul2", params=params_classify, credentials=creds)
    langchain_model_classify = LangChainInterface(model="google/flan-ul2", params=params_classify, credentials=creds)

    params_summary = GenerateParams(decoding_method="greedy", repetition_penalty=2, min_new_tokens=80, max_new_tokens=200)
    langchain_model_summary = LangChainInterface(model="google/flan-ul2", params=params_summary, credentials=creds)
 
    st.set_page_config("Negative News Screening Application", page_icon="")
    st.header("Negative News Screening Dashboard")
    st.caption(":blue[Screening Source: Google Search]")
    st.write('Click below to generate Negative News Screening Results')
    
    with st.sidebar:
        st.sidebar.title("Watsonx.ai (Gen AI) based Negative News Screening App")
        st.subheader("Search Inputs:")
        
        screening_source = st.selectbox('Select the screening source',('Google Search',))
        
        subject_name = st.selectbox("Enter the Name",('James Alexander','Bank Alfalah','Mahinda Rajapaksa','Michael Jackson','Mehul Choksi',))

        if subject_name == "Bank Alfalah":
            entity = "Y"
        else:
            entity = "N"

        #search_keywords = st.text_input("Enter the search keywords")
        options = st.multiselect(
            "Enter the search keywords",
            ["Crime","Terroris", "Terrorism financing", "criminal proceedings", "imprisonment",  "Rape", "Arrest", "Lawsuit", "Sexual abuse", "stock manipulation","money laundering","jailed","warrant","financial crime","fraud", "corruption", "human rights violations", "bankruptcy", "legal proceedings","regulatory penalty","drug trafficking","arms dealing","other illegal activities"],
            default=["Crime","Terroris"],
            label_visibility = 'hidden'
            )
        #print(options)
        if len(options) == 0:
            query = subject_name
        else:    
            search_keywords = " OR ".join(options)
            query = subject_name + " AND " + search_keywords
        
        #print(query)

        st.subheader("Details To Identify True Hit")
        filter_param = []
        location_ip = st.text_input("Enter the Residence or Location")
        #dob_ip = st.text_input("Enter the DOB/DOI")

        with st.expander('DOB/DOI'):
            this_year = datetime.now().today().year
            this_month = datetime.now().today().month
            report_year = st.selectbox('', range(this_year, this_year - 99, -1))
            month_abbr = calendar.month_abbr[1:]
            report_month_str = st.radio('', month_abbr, index=this_month - 1, horizontal=True)
            #report_month = month_abbr.index(report_month_str) + 1
            dob_ip = report_month_str+" "+str(report_year)
        #st.text(f'{report_month_str} {report_year}')

        #print(dob_ip)
        
        if location_ip == "":
            filter_param=[]
        else:
            filter_param.append(location_ip)
            filter_param.append(dob_ip)
        
        write_list("filter.json", filter_param)

        num_results = st.sidebar.slider("Select number of Top search results to analyze", 1, 5, value=5)        
        num_results = int(num_results)

        #st.write("")
        #st.write("")
        #st.write("")

        #char_size = st.sidebar.slider("Set ~7000 to remain within context limit. Reduce only if any Token Limit related Error occurred", 2000, 15000,value=7000)        
        char_size= 12000
        
        if st.button('Process'):
            with st.spinner("Processing"):
                print("")
                print("Start")
                #data = search_func(query, num_results,api_key)
                #print("step1")
                #valid_url_details, bad_url_details = validate_urls(data)
                #print("step2")
                #report_bad_urls(bad_url_details)
                #print("step3")
                #scraped_news = scrape_func(valid_url_details, char_size)
                #print("step4")
                
                neg_news, pos_news =  check_neg_news(num_results, subject_name, langchain_model_classify,genai_model)
                print("step1")
                report_pos_news(pos_news,langchain_model_summary,subject_name, langchain_model_classify)
                print("step2")
                tp,fp = apply_filters(neg_news,langchain_model_classify,subject_name,entity)
                print("step3")
                report_fp(fp,langchain_model_summary,subject_name)
                print("step4")
                report_tp(tp,langchain_model_summary,subject_name)
                print("step5")
                final_conclusion(tp,fp, pos_news, subject_name, num_results)
                print("step6")
                
                print("End")
                print("")
                st.success("Done!")
    
    if st.button('Generate'):
        with st.spinner("Processing"):
            r_tp = read_list("tp_results.json")
            if len(r_tp) > 0:
                df_tp = pd.DataFrame(r_tp, columns =['Url', 'Published Date','Summary','Topic','name_match','is_accused','residence_match','dob_match','age_match'])
                df_tp['News Link'] = df_tp['Url'].apply(make_clickable)
                #df_tp_small = df_tp[['News Link','Published Date']]
                df_tp["Risk Score"] = df_tp.apply(f1_rank_order, axis=1)
                df_tp["Topics"] = df_tp.apply(f1_list2str, axis=1)
                #print(df_tp)
                df_tp = df_tp.sort_values(by=['Risk Score'], ascending=False)
                df_tp.reset_index(drop = True, inplace = True)
                df_tp['Rank'] = df_tp.index+1
                df_tp_small = df_tp[['News Link','Published Date']]
                df_tp = df_tp[['Rank','Summary','Topics','Risk Score']]

            r_fp = read_list("fp_results.json")
            if len(r_fp) > 0:
                df_fp = pd.DataFrame(r_fp, columns =['Url', 'Published Date','Summary','Topic','name_match','is_accused','residence_match','dob_match','age_match'])
                df_fp['News Link'] = df_fp['Url'].apply(make_clickable)
                df_fp["False +ve Reason"] = df_fp.apply(calc_new_col, axis=1)
                #df_fp_small = df_fp[['News Link','Published Date','False +ve Reason']]
                df_fp["Risk Score"] = df_fp.apply(f1_rank_order, axis=1)
                df_fp["Topics"] = df_fp.apply(f1_list2str, axis=1)
                #print(df_fp)
                df_fp = df_fp.sort_values(by=['Risk Score'], ascending=False)
                df_fp.reset_index(drop = True, inplace = True)
                df_fp['Rank'] = df_fp.index+1
                df_fp_small = df_fp[['News Link','Published Date','False +ve Reason']]
                df_fp = df_fp[['Rank','Summary','Topics','Risk Score']]

            r_pos_news = read_list("pos_news_results.json")
            if len(r_pos_news) > 0:
                df_pos_news = pd.DataFrame(r_pos_news, columns =['Url', 'Published Date','Summary'])
                df_pos_news = df_pos_news[['Url','Published Date','Summary']]
                df_pos_news['News Link'] = df_pos_news['Url'].apply(make_clickable)
                df_pos_news = df_pos_news[['News Link','Published Date','Summary']]

            r_bad_urls = read_list("bad_url.json")
            if len(r_bad_urls) > 0:
                df_bad_url = pd.DataFrame(r_bad_urls, columns =['Title', 'Url', 'Snippet', 'Published Date','Failure Reason'])
                df_bad_url = df_bad_url[['Url','Failure Reason']]
                df_bad_url['News Link'] = df_bad_url['Url'].apply(make_clickable)
                df_bad_url = df_bad_url[['News Link','Failure Reason']]      

            #tab1, tab2, tab3, tab4, tab5 = st.tabs(["Negative News", "Un-related results", "Non-Negative news", "Bad URLs", "Screening Conclusion"])
            tab1, tab2, tab3, tab5 = st.tabs(["Negative News", "Un-related results","Non-Negative news", "Screening Conclusion"])
            r_conclusion = read_list("neg_news_conclusion.json")
            conclusion_text1 = r_conclusion[0]
            conclusion_text2 = r_conclusion[1]
            conclusion_text3 = r_conclusion[2]
            conclusion_text4 = r_conclusion[3]

            # style
            th_props = [
            ('font-size', '16px'),
            ('text-align', 'center'),
            ('font-weight', 'bold'),
            ('border', '2px solid white'),
            ('padding', '10px'),
            ('color', '#ffffff'),                                                
            ('background-color', '#98a621')                             
            ]

            th_props_summ = [
            ('font-size', '16px'),
            ('text-align', 'center'),
            ('font-weight', 'bold'),
            ('border', '2px solid white'),
            ('padding', '10px'),
            ('color', '#ffffff'),                                                
            ('background-color', '#030e1a')                             
            ]

            td_props = [
            ('font-size', '14px'),
            ('border', '1px solid grey'),
            ('padding', '10px'),
            ('color', '#000000'),                      
            #('background-color', '#dcdcdc')             
            ]

            td_props_summ = [
            ('font-size', '14px'),
            ('border', '1px solid grey'),
            ('padding', '10px'),
            ('font-weight', 'bold'),
            ('color', '#6A7309'),                      
            #('background-color', '#dcdcdc')             
            ]

            tr_odd = [
            ('background-color', '#fcf4e6')
            ]
            
            tr_even = [
            ('background-color', '#e6ded1')
            ]

            tr_odd_summ = [
            ('background-color', '#dcdcdc')
            ]
            
            tr_even_summ = [
            ('background-color', '#b3afaf')
            ]

            styles = [
            dict(selector="th", props=th_props),
            dict(selector="td", props=td_props),
            #dict(selector="tr:nth-child(odd)", props=tr_odd),
            dict(selector="tr:nth-child(even)", props=tr_even)
            ]

            styles_summ = [
            dict(selector="th", props=th_props_summ),
            dict(selector="td", props=td_props_summ),
            dict(selector="tr:nth-child(odd)", props=tr_odd_summ),
            dict(selector="tr:nth-child(even)", props=tr_even_summ)
            ]

            with tab1:
                st.subheader("Negative News", divider='rainbow')
                if len(r_tp) > 0:
                    st.write(df_tp_small.style.set_properties(**{'text-align': 'left'}).set_table_styles(styles).hide(axis=0).to_html(escape = False), unsafe_allow_html = True)
                    st.subheader('News Summary', divider='rainbow')
                    st.write(df_tp.style.set_properties(**{'text-align': 'left'}).set_table_styles(styles_summ).hide(axis=0).to_html(escape = False), unsafe_allow_html = True)
                else:
                    st.write(":blue[There is no Negative News alert for "+subject_name+".]")     

            with tab2:
                st.subheader("Un-related results", divider='rainbow')
                if len(r_fp) > 0:
                    st.write(df_fp_small.style.set_properties(**{'text-align': 'left'}).set_table_styles(styles).hide(axis=0).to_html(escape = False), unsafe_allow_html = True)
                    st.subheader('News Summary', divider='rainbow')
                    st.write(df_fp.style.set_properties(**{'text-align': 'left'}).set_table_styles(styles_summ).hide(axis=0).to_html(escape = False), unsafe_allow_html = True)
                else:
                    st.write(":blue[There is no Un-related result found.]")
            
            with tab3:
                st.subheader("Non-Negative news", divider='rainbow')
                if len(r_pos_news) > 0:
                    st.markdown(df_pos_news.style.set_properties(**{'text-align': 'left'}).set_table_styles(styles_summ).hide(axis=0).to_html(escape = False), unsafe_allow_html = True)
                else:
                    st.write(":green[There is no Non-Negative news present.]")

            with tab5:
                st.subheader("Screening Conclusion", divider='rainbow')
                st.write(':green['+conclusion_text1+']')
                st.write(':blue['+conclusion_text4+']')
                st.write(':green['+conclusion_text2+']')
                st.write(':green['+conclusion_text3+']')
            
            #with tab4:
                #st.subheader("Bad URLs (for manual check)", divider='rainbow')
                #if len(r_bad_urls) > 0:
                    #st.markdown(df_bad_url.style.set_properties(**{'text-align': 'left'}).set_table_styles(styles).hide(axis=0).to_html(escape = False), unsafe_allow_html = True)
                #else:
                    #st.write(":green[There is no bad url present.]")

if __name__ ==  '__main__':
    main()
