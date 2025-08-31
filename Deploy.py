
# coding: utf-8

# In[1]:


import re
import PyPDF2
import docx2txt
import pdfplumber
import pandas as pd
import streamlit as st
import warnings
warnings.filterwarnings('ignore')


# In[2]:


import en_core_web_sm
nlp = en_core_web_sm.load()
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer


# In[3]:


st.title('RESUME CLASSIFICATION')
st.markdown('<style>h1{color:Blue;}</style>', unsafe_allow_html=True)
st.subheader('Welcome to Resume Classification App')


# In[4]:


def extract_skills(resume_text):
    nlp_text = nlp(resume_text)
    noun_chunks = nlp_text.noun_chunks
    tokens = [token.text for token in nlp_text if not token.is_stop] # removing stop words and implementing word tokenization

    data = pd.read_csv(r"skills.csv") # reading the csv file
    skills = list(data.columns.values) # extract values       
    skillset = []

    for token in tokens: # check for one-grams (example: python)
        if token.lower() in skills:
            skillset.append(token)

    for token in noun_chunks: # check for bi-grams and tri-grams (example: machine learning)
        token = token.text.lower().strip()
        if token in skills:
            skillset.append(token)   
    return [i.capitalize() for i in set([i.lower() for i in skillset])]


# In[5]:


def getText(filename):
    fullText = '' # Create empty string 
    if filename.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx2txt.process(filename)
        for para in doc:
            fullText = fullText + para      
    else:  
        with pdfplumber.open(filename) as pdf_file:
            pdoc = PyPDF2.PdfFileReader(filename)
            number_of_pages = pdoc.getNumPages()
            page = pdoc.pages[0]
            page_content = page.extractText()
        for paragraph in page_content:
            fullText =  fullText + paragraph       
    return (fullText)


# In[6]:


def display(doc_file):
    resume = []
    if doc_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        resume.append(docx2txt.process(doc_file))
    else:
        with pdfplumber.open(doc_file) as pdf:
            pages=pdf.pages[0]
            resume.append(pages.extract_text())  
    return resume


# In[7]:


def preprocess(sentence):
    sentence = str(sentence)
    sentence = sentence.lower()
    sentence = sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url = re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w) for w in filtered_words]
    return " ".join(lemma_words) 

file_type=pd.DataFrame([], columns=['Uploaded File',  'Predicted Profile','Skills',])
filename = []
predicted = []
skills = []


# In[8]:


import pickle as pk
model = pk.load(open(r'modelSVM.pkl', 'rb'))
Vectorizer = pk.load(open(r'vector.pkl', 'rb'))


# In[9]:


upload_file = st.file_uploader('Upload Your Resumes', type= ['docx','pdf'],accept_multiple_files=True)


# In[12]:


for doc_file in upload_file:
    if doc_file is not None:
        filename.append(doc_file.name)
        cleaned = preprocess(display(doc_file))
        prediction = model.predict(Vectorizer.transform([cleaned]))[0]
        predicted.append(prediction)
        extText = getText(doc_file)
        skills.append(extract_skills(extText))
        
if len(predicted) > 0:
    file_type['Uploaded File'] = filename
    file_type['Skills'] = skills
    file_type['Predicted Profile'] = predicted
    st.table(file_type.style.format())
    
select = ['Peoplesoft Resume','SQL Developer','React Developer','workday']
st.subheader('Select as per Requirement')
option = st.selectbox('Fields',select)

if option == 'Peoplesoft Resume':
    st.table(file_type[file_type['Predicted Profile'] == 'Peoplesoft Resume'])
elif option == 'SQL Developer':
    st.table(file_type[file_type['Predicted Profile'] == 'SQL Developer'])
elif option == 'React Developer':
    st.table(file_type[file_type['Predicted Profile'] == 'React Developer'])
elif option == 'workday':
    st.table(file_type[file_type['Predicted Profile'] == 'workday'])


# In[ ]:




