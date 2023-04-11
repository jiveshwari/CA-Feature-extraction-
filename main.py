import pandas as pd
import gensim 
import numpy as np
from sklearn import preprocessing,svm
from sklearn.metrics import accuracy_score,classification_report
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk import pos_tag
from sklearn.naive_bayes import GaussianNB

model = gensim.models.Word2Vec.load('wor2vec_bio_labels.model')
    
def buildWord2Vec(sentences):
    model = gensim.models.Word2Vec(min_count=1,window=10,vector_size=1)
    corpus_iterable = [word_tokenize(i) for i in sentences]
    model.build_vocab(corpus_iterable=corpus_iterable)
    model.train(corpus_iterable=corpus_iterable,total_examples=len(corpus_iterable),epochs=model.epochs)
    model.save('wor2vec_bio_labels.model')

def svm_model():
    SVM = svm.SVC(C=1.0, kernel='rbf', gamma='auto',verbose=1)
    return SVM 

def naivebayes():
    gnb = GaussianNB()
    return gnb

def word2vec(word):
    try:
        return model.wv[word]
    except KeyError:
        default_vector = [0.00000000] 
        return default_vector

def sent2vec(sent):
    words = word_tokenize(sent)
    v = []
    for w in words:
        try:
            v.append(word2vec(w))
        except:
            continue
    return np.array(v).mean(axis=0)

def main():
    
    train_path = './data/train-bio.csv'
    test_path = './data/test-bio.csv'
    
    train = [x.split('\t') for x in open(train_path).read().split('\n')]
    train = [x for x in train if len(x) > 1]
    redundant_tokens = ['__END_PARAGRAPH__',  '__END_ESSAY__']
    train = [x for x in train if x[0] not in redundant_tokens]
    trainTokens = [t[0] for t in train]
    trainLabels = [t[1] for t in train]
    trainCorpus = ' '.join(trainTokens)
    trainSentenceCorpus = sent_tokenize(trainCorpus)

    test = [x.split('\t') for x in open(test_path).read().split('\n')]
    test = [x for x in test if len(x) > 1]
    test = [x for x in test if x[0] not in redundant_tokens]
    testTokens = [t[0] for t in test]
    testLabels = [t[1] for t in test]
    testCorpus = ' '.join(testTokens)
    testSentenceCorpus = sent_tokenize(testCorpus)

    trainPOS = [i[1] for i in pos_tag(trainCorpus.split(' '))]
    testPOS = [i[1] for i in pos_tag(testCorpus.split(' '))]
    
    trainSentence = []
    for token in range(0,len(trainTokens)):
        try:
            sentLen = len(word_tokenize(trainSentenceCorpus[token])) 
            while sentLen:
                trainSentence.append(trainSentenceCorpus[token])
                sentLen = sentLen - 1
        except IndexError:
            break
    for tok in range(len(trainSentence),len(trainTokens)):
        word = trainTokens[tok]
        for sent in trainSentenceCorpus:
            if word in sent:
                trainSentence.append(sent)
                break      
    
    # print("trainTokens: ",len(trainTokens))
    # print("trainPOS: ",len(trainPOS))
    # print("trainSentence: ",len(trainSentence))
    
    testSentence = []
    for token in range(0,len(testTokens)):
        try:
            sentLen = len(word_tokenize(testSentenceCorpus[token])) 
            while sentLen:
                testSentence.append(testSentenceCorpus[token])
                sentLen = sentLen - 1
        except IndexError:
            break
    for tok in range(len(testSentence),len(testTokens)):
        word = testTokens[tok]
        for sent in testSentenceCorpus:
            if word in sent:
                testSentence.append(sent)
                break

    # print("testTokens: ",len(testTokens))
    # print("testPOS: ",len(testPOS))
    # print("testSentence: ",len(testSentence))
    
    # buildWord2Vec(list(trainSentenceCorpus+testSentenceCorpus+list(set(trainPOS))+list(set(testPOS))))
    
    
    trainUnitPosition = [] 
    for idx,sent in enumerate(trainSentence):
        try:
            trainUnitPosition.append(sent.index(trainTokens[idx]))
        except ValueError:
            trainUnitPosition.append(len(sent))

    testUnitPosition = [] 
    for idx,sent in enumerate(testSentence):
        try:
            testUnitPosition.append(sent.index(testTokens[idx]))
        except ValueError:
            testUnitPosition.append(len(sent))

    df = pd.DataFrame({'tokens':trainTokens,'pos': trainPOS,'sentences':trainSentence,'unit_pos':trainUnitPosition })
    testDf = pd.DataFrame({'tokens':testTokens,'pos':testPOS,'sentences':testSentence,'unit_pos':testUnitPosition})
    
    df['token_len'] = df['tokens'].apply(lambda x: np.array(len(str(x))))
    df['token_freq'] = df.groupby('tokens')['tokens'].transform('count')
    df['tokens'] = df['tokens'].apply(lambda x: np.array(word2vec(x)))
    df['pos'] = df['pos'].apply(lambda x: np.array(word2vec(x)))
    df['sent_len'] = df['sentences'].apply(lambda x: np.array(len(str(x))))
    df['words_in_sent'] = df['sentences'].apply(lambda x: np.array(len(word_tokenize(str(x)))))
    df['sentences'] = df['sentences'].apply(lambda x: np.array(sent2vec(x)))
    
    testDf['token_len'] = testDf['tokens'].apply(lambda x: np.array(len(str(x))))
    testDf['token_freq'] = testDf.groupby('tokens')['tokens'].transform('count')
    testDf['tokens'] = testDf['tokens'].apply(lambda x: np.array(word2vec(x)))
    testDf['pos'] = testDf['pos'].apply(lambda x: np.array(word2vec(x)))
    testDf['sent_len'] = testDf['sentences'].apply(lambda x: np.array(len(str(x))))
    testDf['words_in_sent'] = testDf['sentences'].apply(lambda x: np.array(len(word_tokenize(str(x)))))
    testDf['sentences'] = testDf['sentences'].apply(lambda x: np.array(sent2vec(x)))
    
    x_train = df[['tokens', 'pos','token_len','token_freq','sentences','words_in_sent','sent_len','unit_pos']]
    x_test = testDf[['tokens', 'pos','token_len','token_freq','sentences','words_in_sent','sent_len','unit_pos']]
    
    lb = preprocessing.LabelEncoder()
    y_train = lb.fit_transform(trainLabels)
    y_test = lb.fit_transform(testLabels)
    
    # print(x_train)
    x_train = preprocessing.scale(x_train)
    x_test = preprocessing.scale(x_test)
    
    svm = svm_model()
    svm.fit(x_train,y_train)
    predictions_SVM = svm.predict(x_test)
    print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, y_test)*100)
    print("\nreport: ",classification_report(y_test,predictions_SVM))
      
    # nb = naivebayes()
    # nb.fit(x_train, y_train)
    # y_pred = nb.predict(x_test)
    # print("NB Accuracy Score -> ",accuracy_score(y_pred, y_test)*100)
    # print("\nreport: ",classification_report(y_test,y_pred))
    
    # print("x_train",x_train)
    # print("x_train.shape",x_train.shape)
    # print("x_test",x_test)
    # print("x_test.shape",x_test.shape)
    # print("y_train",y_train)
    # print("y_train.shape",y_train.shape)
    # print("y_test",y_test)
    # print("y_test.shape",y_test.shape)
    # print("y_pred_decoded: ",y_pred_decoded)
    
    output = {
        'tokens': [],
        'labels': []
    }
    y_pred_decoded = lb.inverse_transform(predictions_SVM)
    for p in range(len(predictions_SVM)):
        output['tokens'].append(str(testTokens[p]))
        output['labels'].append(str(y_pred_decoded[p]))
    
    o = pd.DataFrame(output)
    print(o.head())
    o.to_csv('prediction.csv',header=False,index=False,sep='\t')
    

if __name__ == '__main__':
    main()

# import tensorflow as tf
# from tensorflow.keras.layers import Embedding
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.preprocessing.text import one_hot
# from tensorflow.keras.layers import LSTM
# from tensorflow.keras.layers import Bidirectional
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.utils import to_categorical
# from tensorflow import keras
# from keras import layers
# from sklearn.base import clone
# from sklearn.compose import make_column_transformer
# from sklearn.metrics import accuracy_score
# from sklearn.feature_extraction import DictVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction import DictVectorizer
# nltk.download('averaged_perceptron_tagger')
# from sklearn.metrics import confusion_matrix
# import nltk
# nltk.download('punkt')

# def nueral_network(input_dim):
#     #Creating model
#     # embedding_vector_features=10
#     model=Sequential()
#     # model.add(Embedding(voc_size,input_dim,input_length=sent_length))
#     model.add(Dense(10, input_dim=input_dim, activation='relu'))
#     model.add(Dense(3, activation='softmax'))
#     # model.add(Bidirectional(LSTM(100)))
#     # model.add(Dense(1,activation='softmax'))
#     model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#     return model
    # model.save('bilstm')
#     # print(model.summary())
