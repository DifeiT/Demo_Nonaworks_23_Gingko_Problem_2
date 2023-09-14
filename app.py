# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 16:21:25 2022

@author: luol2
"""

import streamlit as st
from src.nn_model import bioTag_CNN,bioTag_Bioformer
from src.dic_ner import dic_ont
from src.tagging_text import bioTag
import os
import json
from pandas import DataFrame
import nltk 
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

st.set_page_config(
    page_title="Demo",
    page_icon="🎈",
    layout="wide",
    menu_items={
    }
)


# def _max_width_():
#     max_width_str = f"max-width: 2400px;"
#     st.markdown(
#         f"""
#     <style>
#     .reportview-container .main .block-container{{
#         {max_width_str}
#     }}
#     </style>    
#     """,
#         unsafe_allow_html=True,
#     )


# _max_width_()

# c30, c31, c32 = st.columns([2.5, 1, 3])

# with c30:
#     # st.image("logo.png", width=400)
st.title("Demo")

with st.expander("🎈 About this demo", expanded=True):

    st.write(
        """     
-   This demo is an extension work using [PhenoTagger](https://github.com/ncbi-nlp/PhenoTagger) library
-   Hackathon: Nonaworks - Gingko Problem 2
	    """
    )

    st.markdown("")

st.markdown("")
st.markdown("## ✂ Paste your text ")
with st.form(key="my_form"):


    ce, c1, ce, c2, c3 = st.columns([0.07, 1, 0.07, 4, 0.07])
    with c1:
        ModelType = st.radio(
            "Choose your model",
            ["Bioformer(Default)"],
            help="Bioformer is more precise, CNN is more efficient",
        )

        if ModelType == "Bioformer(Default)":
            # kw_model = KeyBERT(model=roberta)

            @st.cache(allow_output_mutation=True)
            def load_model(model='hpo'):
                if model =='1':
                    ontfiles = {'dic_file': './dict_new_fyeco/noabb_lemma.dic',
                                'word_hpo_file': './dict_new_fyeco/word_id_map.json',
                                'hpo_word_file': './dict_new_fyeco/id_word_map.json'}

                    vocabfiles = {'labelfile': './dict_new_fyeco/lable.vocab',
                                  'config_path': './vocab/bioformer-cased-v1.0/bert_config.json',
                                  'checkpoint_path': './vocab/bioformer-cased-v1.0/bioformer-cased-v1.0-model.ckpt-2000000',
                                  'vocab_path': './vocab/bioformer-cased-v1.0/vocab.txt'}

                    modelfile = './vocab/bioformer_fyeco.h5'

                elif model == '2':
                    vocabfiles = {'labelfile': './dict_new_hpo/lable.vocab',
                                  'config_path': './vocab/bioformer-cased-v1.0/bert_config.json',
                                  'checkpoint_path': './vocab/bioformer-cased-v1.0/bioformer-cased-v1.0-model.ckpt-2000000',
                                  'vocab_path': './vocab/bioformer-cased-v1.0/vocab.txt'}

                    ontfiles = {'dic_file': './dict_new_hpo/noabb_lemma.dic',
                                'word_hpo_file': './dict_new_hpo/word_id_map.json',
                                'hpo_word_file': './dict_new_hpo/id_word_map.json'}

                    modelfile='./vocab/bioformer_p5n5_b64_1e-5_95_hponew3.h5'

                elif model == '3':
                    vocabfiles = {'labelfile': './dict_new_sympo/lable.vocab',
                                  'config_path': './vocab/bioformer-cased-v1.0/bert_config.json',
                                  'checkpoint_path': './vocab/bioformer-cased-v1.0/bioformer-cased-v1.0-model.ckpt-2000000',
                                  'vocab_path': './vocab/bioformer-cased-v1.0/vocab.txt'}

                    ontfiles = {'dic_file': './dict_new_sympo/noabb_lemma.dic',
                                'word_hpo_file': './dict_new_sympo/word_id_map.json',
                                'hpo_word_file': './dict_new_sympo/id_word_map.json'}

                    modelfile='./vocab/bioformer_sympo.h5'
                    pass

                biotag_dic=dic_ont(ontfiles)    
    
                nn_model=bioTag_Bioformer(vocabfiles)
                nn_model.load_model(modelfile)
                return nn_model,biotag_dic

            nn_model1, biotag_dic1 = load_model(model='1')
            nn_model2, biotag_dic2 = load_model(model='2')
            nn_model3, biotag_dic3 = load_model(model='3')

        else:
            @st.cache(allow_output_mutation=True)
            def load_model():
                ontfiles={'dic_file':'./dict_new/noabb_lemma.dic',
                  'word_hpo_file':'./dict_new/word_id_map.json',
                  'hpo_word_file':'./dict_new/id_word_map.json'}
        

                vocabfiles={'w2vfile':'./vocab/bio_embedding_intrinsic.d200',   
                            'charfile':'./vocab/char.vocab',
                            'labelfile':'./dict_new/lable.vocab',
                            'posfile':'./vocab/pos.vocab'}
                modelfile='./vocab/cnn_p5n5_b128_95_hponew1.h5'
        
                biotag_dic=dic_ont(ontfiles)    
            
                nn_model=bioTag_CNN(vocabfiles)
                nn_model.load_model(modelfile)
            
                return nn_model,biotag_dic

            nn_model,biotag_dic = load_model()
        
        para_overlap = st.checkbox(
            "Overlap concept",
            value=False,
            help="Tick this box to identify overlapping concepts",
        )
        para_abbr = st.checkbox(
            "Abbreviaitons",
            value=True,
            help="Tick this box to identify abbreviations",
        )        
        
        para_threshold = st.slider(
            "Threshold",
            min_value=0.5,
            max_value=1.0,
            value=0.95,
            step=0.05,
            help="Retrun the preditions which socre over the threshold.",
        )
      



    with c2:
        
            
        doc = st.text_area(
              "Paste your text below",
              value = 'The clinical features of Angelman syndrome (AS) comprise severe mental retardation, postnatal microcephaly, macrostomia and prognathia, absence of speech, ataxia, and a happy disposition. We report on seven patients who lack most of these features, but presented with obesity, muscular hypotonia and mild mental retardation. Based on the latter findings, the patients were initially suspected of having Prader-Willi syndrome. DNA methylation analysis of SNRPN and D15S63, however, revealed an AS pattern, ie the maternal band was faint or absent. Cytogenetic studies and microsatellite analysis demonstrated apparently normal chromosomes 15 of biparental inheritance. We conclude that these patients have an imprinting defect and a previously unrecognised form of AS. The mild phenotype may be explained by an incomplete imprinting defect or by cellular mosaicism.',
              height=400,
        )
        

        # MAX_WORDS = 500
        # import re
        # res = len(re.findall(r"\w+", doc))
        # if res > MAX_WORDS:
        #     st.warning(
        #         "⚠️ Your text contains "
        #         + str(res)
        #         + " words."
        #         + " Only the first 500 words will be reviewed. Stay tuned as increased allowance is coming! 😊"
        #     )

        #     doc = doc[:MAX_WORDS]

        submit_button = st.form_submit_button(label="🖱️ Submit!")


if not submit_button:
    st.stop()

#st.write(para_overlap,para_abbr,para_threshold)
para_set={
          #model_type':para_model, # cnn or bioformer
          'onlyLongest': not para_overlap, # False: return overlap concepts, True only longgest
          'abbrRecog':para_abbr,# False: don't identify abbr, True: identify abbr
          'ML_Threshold':para_threshold,# the Threshold of deep learning model
          }
st.markdown("")
st.markdown("## ⏳ Tagging results:")
with st.spinner('Wait for tagging...'):
    tag_result1=bioTag(doc,biotag_dic1,nn_model1,onlyLongest=para_set['onlyLongest'], abbrRecog=para_set['abbrRecog'],Threshold=para_set['ML_Threshold'])
    tag_result2=bioTag(doc,biotag_dic2,nn_model2,onlyLongest=para_set['onlyLongest'], abbrRecog=para_set['abbrRecog'],Threshold=para_set['ML_Threshold'])
    tag_result3=bioTag(doc,biotag_dic3,nn_model3,onlyLongest=para_set['onlyLongest'], abbrRecog=para_set['abbrRecog'],Threshold=para_set['ML_Threshold'])

st.markdown('<font style="color: rgb(128, 128, 128);">Move the mouse over the entity to display the id.</font>', unsafe_allow_html=True)
# print('dic...........:',biotag_dic.keys())
# st.write('parameters:', para_overlap,para_abbr,para_threshold)

html_results=''
text_results=doc+'\n'
entity_end=0

# poid_counts = []

hpoid_count1= {}
hpoid_count2 = {}
hpoid_count3 = {}

tag_display = {}

flag = False
if len(tag_result1)>=0:
    for ele in tag_result1:
        entity_start=int(ele[0])
        #html_results+=doc[entity_end:entity_start]
        entity_end=int(ele[1])
        entity_id=ele[2]
        entity_score=ele[3]
        tag_display[entity_start] = (entity_end, entity_id, "1")
        text_results+=ele[0]+'\t'+ele[1]+'\t'+doc[entity_start:entity_end]+'\t'+ele[2]+'\t'+format(float(ele[3]),'.2f')+'\n'

        if entity_id not in hpoid_count1.keys():
            hpoid_count1[entity_id]=1
        else:
            hpoid_count1[entity_id]+=1
        
        #html_results+='<font style="background-color: rgb(255, 204, 0)'+';" title="'+entity_id+'">'+doc[entity_start:entity_end]+'</font>'
    #html_results+=doc[entity_end:]

    flag = True

if len(tag_result2) >= 0:
    entity_end = 0
    for ele in tag_result2:
        entity_start = int(ele[0])
        #html_results += doc[entity_end:entity_start]
        entity_end = int(ele[1])
        entity_id = ele[2]
        entity_score = ele[3]
        tag_display[entity_start] = (entity_end, entity_id, "2")
        text_results += ele[0] + '\t' + ele[1] + '\t' + doc[entity_start:entity_end] + '\t' + ele[2] + '\t' + format(
            float(ele[3]), '.2f') + '\n'

        if entity_id not in hpoid_count2.keys():
            hpoid_count2[entity_id] = 1
        else:
            hpoid_count2[entity_id] += 1

       # html_results += '<font style="background-color: rgb(255, 0, 0)' + ';" title="' + entity_id + '">' + doc[entity_start:entity_end] + '</font>'
    #html_results += doc[entity_end:]

    flag = True

if len(tag_result3) >= 0:
    entity_end = 0
    for ele in tag_result3:
        entity_start = int(ele[0])
        #html_results += doc[entity_end:entity_start]
        entity_end = int(ele[1])
        entity_id = ele[2]
        entity_score = ele[3]
        tag_display[entity_start] = (entity_end, entity_id, "3")
        text_results += ele[0] + '\t' + ele[1] + '\t' + doc[entity_start:entity_end] + '\t' + ele[2] + '\t' + format(
            float(ele[3]), '.2f') + '\n'

        if entity_id not in hpoid_count3.keys():
            hpoid_count3[entity_id] = 1
        else:
            hpoid_count3[entity_id] += 1

       # html_results += '<font style="background-color: rgb(255, 0, 0)' + ';" title="' + entity_id + '">' + doc[entity_start:entity_end] + '</font>'
    #html_results += doc[entity_end:]

    flag = True

if not flag:
    html_results = doc
else:
    myKeys = list(tag_display.keys())
    myKeys.sort()
    sorted_tag_display = {i: tag_display[i] for i in myKeys}
    entity_end = 0

    for entity_start, value in sorted_tag_display.items():
        html_results += doc[entity_end:entity_start]
        entity_end = value[0]
        entity_id = value[1]
        type = value[2]
        if type == "1":
            html_results += '<font style="background-color: rgb(255, 204, 0)' + ';" title="' + entity_id + '">' + doc[entity_start:entity_end] + '</font>'
        elif type == "2":
            html_results += '<font style="background-color: rgb(255, 0, 0)' + ';" title="' + entity_id + '">' + doc[entity_start:entity_end] + '</font>'
        elif type == "3":
            html_results += '<font style="background-color: rgb(102, 255, 178)' + ';" title="' + entity_id + '">' + doc[entity_start:entity_end] + '</font>'

    html_results += doc[entity_end:]
    
    st.markdown('<table border="1"><tr><td>'+html_results+'</td></tr></table>', unsafe_allow_html=True)

#table
data_entity=[]
for ele in hpoid_count1.keys():
    segs=ele.split(';')
    term_name=''
    for seg in segs:
        term_name+=biotag_dic1.hpo_word[seg][0]+';'
    temp=[ele,term_name,hpoid_count1[ele]] #hpoid, term name, count
    data_entity.append(temp)

for ele in hpoid_count2.keys():
    segs=ele.split(';')
    term_name=''
    for seg in segs:
        term_name+=biotag_dic2.hpo_word[seg][0]+';'
    temp=[ele,term_name,hpoid_count2[ele]] #hpoid, term name, count
    data_entity.append(temp)

for ele in hpoid_count3.keys():
    segs=ele.split(';')
    term_name=''
    for seg in segs:
        term_name+=biotag_dic3.hpo_word[seg][0]+';'
    temp=[ele,term_name,hpoid_count3[ele]] #hpoid, term name, count
    data_entity.append(temp)



st.markdown("")
st.markdown("")
# st.markdown("## Table output:")

# cs, c1, c2, c3, cLast = st.columns([2, 1.5, 1.5, 1.5, 2])

# with c1:
#     CSVButton2 = download_button(keywords, "Data.csv", "📥 Download (.csv)")
# with c2:
#     CSVButton2 = download_button(keywords, "Data.txt", "📥 Download (.txt)")
# with c3:
#     CSVButton2 = download_button(keywords, "Data.json", "📥 Download (.json)")

# st.header("")

df = (
    DataFrame(data_entity, columns=["Phenotype ID", "Term Name","Frequency"])
    .sort_values(by="Frequency", ascending=False)
    .reset_index(drop=True)
)

df.index += 1

c1, c2, c3 = st.columns([1, 4, 1])

# format_dictionary = {
#     "Relevancy": "{:.1%}",
# }

# df = df.format(format_dictionary)

with c2:
    st.table(df)
    
c1, c2, c3 = st.columns([1, 1, 1])
with c2:
    st.download_button('Download annotations', text_results) 
    
