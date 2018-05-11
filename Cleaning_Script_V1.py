import pandas as pd
import nltk
import re
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import os


def pre_processing():
    """
    pre_processing function
    :return:
    """
    try:
        wordnet_lemmatizer = WordNetLemmatizer()
        stopwords = set(w.rstrip() for w in open('stopwords.txt'))

        df_train = pd.read_csv('training.1600000.processed.noemoticon.csv',encoding='latin-1', names=["sentiment","tweet_id","date","query","user_id","text_body"])
        df_test = pd.read_csv('testdata.manual.2009.06.14.csv',encoding='latin-1', names=["sentiment","tweet_id","date","query","user_id","text_body"])

        #Function to remove emoticons from text
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   "]+", flags=re.UNICODE)
        def emoji_remover(text):
            text = emoji_pattern.sub(r'', text) # no emoji
            return text

        #Function to remove special charactars from text
        APPOSTOPHES = { "'s" : " is", "'re" : " are", "'t" : " not", "'m" : " am", "'ll" : " will", "'d" : " would", "'ve" : " have"}
        contractions_re = re.compile("|".join(r'(\b%s\b)' % c for c in APPOSTOPHES.keys()))

        def expand_contractions(s, APPOSTOPHES=APPOSTOPHES):
            def replace(match):
                return APPOSTOPHES[match.group(0)]
            return contractions_re.sub(replace, s)

        #Function to clean text body
        def cleaning(text):
            import string
            exclude = set(string.punctuation)
            import re
            # remove new line and digits with regular expression
            text = re.sub(r'\n', '', text)
            text = re.sub(r'\d', '', text)
            # remove patterns matching url format
            url_pattern = r'((http|ftp|https):\/\/)?[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?'
            text = re.sub(url_pattern, ' ', text)
            #word seperater
            text = " ".join(re.findall('[A-Z][^A-Z]*', text))
            # remove non-ascii characters
            text = ''.join(character for character in text if ord(character) < 128)
            # remove punctuations
            text = ''.join(character for character in text if character not in exclude)
            # standardize white space
            text = re.sub(r'\s+', ' ', text)
            # drop capitalization
            text = text.lower()
            #remove white space
            text = text.strip()
            #from textblob import TextBlob
            #spelling correction
            #text = str(TextBlob(text).correct())
            return text

        #Fuction to tekenize, lemmatize and remove stop words from text
        def my_tokenizer(s):
            s = s.lower() # downcase
            tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
            tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
            tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
            tokens = [t for t in tokens if t not in stopwords] # remove stopwords
            return ' '.join(tokens)

        #Deriving clean text column from twitter data
        df_train['Clean_Text_Body'] = df_train['text_body'].apply(lambda x: emoji_remover(str(x)))
        df_test['Clean_Text_Body'] = df_test['text_body'].apply(lambda x: emoji_remover(str(x)))

        print('---- Stage 1 cleaning complete ----')

        df_train['Clean_Text_Body'] = df_train['Clean_Text_Body'].apply(lambda x: expand_contractions(str(x)))
        df_test['Clean_Text_Body'] = df_test['Clean_Text_Body'].apply(lambda x: expand_contractions(str(x)))

        print('---- Stage 2 cleaning complete ----')

        df_train['Clean_Text_Body'] = df_train['Clean_Text_Body'].apply(lambda x: cleaning(str(x)))
        df_test['Clean_Text_Body'] = df_test['Clean_Text_Body'].apply(lambda x: cleaning(str(x)))

        print('---- Stage 3 cleaning complete ----')

        df_train['Clean_Text_Body'] = df_train['Clean_Text_Body'].apply(lambda x: my_tokenizer(str(x)))
        df_test['Clean_Text_Body'] = df_test['Clean_Text_Body'].apply(lambda x: my_tokenizer(str(x)))

        print('---- Stage 4 cleaning complete ----')

        # Replace and remove empty rows
        df_train['Clean_Text_Body'] = df_train['Clean_Text_Body'].replace('', np.nan)
        df_train = df_train.dropna(how='any')

        df_test['Clean_Text_Body'] = df_test['Clean_Text_Body'].replace('', np.nan)
        df_test = df_test.dropna(how='any')

        print('---- Stage 5 cleaning complete ----')

        df_train_clean = df_train[['tweet_id','text_body','Clean_Text_Body','sentiment']]
        df_test_clean = df_test[['tweet_id','text_body','Clean_Text_Body','sentiment']]

        print('---- Cleaning Complete ----')

        df_train_clean.to_csv('Train_Data_cleaned.csv', index=False)
        print('Train Data Saved, Printing First 5 Rows ==', df_train_clean.head())
        print('Length of Train Data ===', len(df_train_clean))
        
        df_test_clean.to_csv('Test_Data_cleaned.csv', index=False)
        print('Test Data Saved, Printing First 5 Rows ==', df_test_clean.head())
        print('Length of Test Data ===', len(df_test_clean))

               

    except Exception as e:
        print('Exception in Running pre_processing, as:: ', e)

if __name__ == "__main__":

    os.chdir(r'C:\Users\Ishpal\Desktop\Sentiment_Analysis')
    pre_processing()
    

    
        

        

        
