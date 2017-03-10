import re
import cPickle
import xlrd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.cross_validation import train_test_split


from sklearn import neighbors
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

stemmer = SnowballStemmer('english')
stopWords = set(stopwords.words('english'))
vectorizer = CountVectorizer(
    analyzer='word', max_features=10000, ngram_range=(1, 3)
)
tfidf_transformer = TfidfTransformer(norm='l1', use_idf=True)


def importSheet(name):
    sheets = xlrd.open_workbook(name)
    sheet = sheets.sheet_by_index(0)
    return sheet


def cleanDescriptions(descriptions, disc=False):
    clean_descriptions = []
    index = 0

    for desc in descriptions:
        desc = re.sub("[^a-zA-Z]", " ", desc)
        stemmed_desc = []
        for word in desc.split():
            stemmed_word = stemmer.stem(word)
            stemmed_desc.append(stemmed_word)
        meaningful_desc = [
            word for word in stemmed_desc if word not in stopWords
        ]
        clean_descriptions.append(" ".join(meaningful_desc))
        index += 1
    return clean_descriptions


def GetDescriptionsFromSheet(sheet):
    descriptions = []
    for index in xrange(1, sheet.nrows):
        description = sheet.cell_value(index, 1)
        descriptions.append(description)
    return descriptions


def transformByTfIdf(X_train, X_test, transformer):
    X_train = transformer.transform(X_train).toarray()
    X_test = transformer.transform(X_test).toarray()
    return X_train, X_test


def transformByVectorizer(X_train, X_test, vectorizer):
    X_train = vectorizer.transform(X_train).toarray()
    X_test = vectorizer.transform(X_test).toarray()
    return X_train, X_test


def vectorize(descriptions):
    vectorized_descriptions = vectorizer.fit_transform(descriptions)
    return vectorized_descriptions.toarray(), vectorizer


def runForest(X_train, X_test, Y_train, Y_test):
    forest = RandomForestClassifier(n_estimators=50, random_state=1)
    forest = forest.fit(X_train, Y_train)
    score = forest.score(X_test, Y_test)
    return score, forest
    
def runGradientBoost(X_train, X_test, Y_train, Y_test):
    gb = GradientBoostingClassifier(
        n_estimators=50,
        max_depth=5,
        subsample=0.5,
        max_features='log2',
        random_state=1
    )
    gb = gb.fit(X_train, Y_train)
    score = gb.score(X_test, Y_test)
    return score,gb

def createLabelArray(qualified, disqualified):
    labels = []
    for q in qualified:
        labels.append(1)
    for d in disqualified:
        labels.append(0)
    return np.array(labels)

qualified_sheet = importSheet('input/qualified.xlsx')
qualified_descriptions = GetDescriptionsFromSheet(qualified_sheet)
qualified_clean_descriptions = cleanDescriptions(qualified_descriptions)
disqualified_sheet = importSheet('input/disqualified.xlsx')
disqualified_descriptions = GetDescriptionsFromSheet(disqualified_sheet)
disqualified_clean_descriptions = cleanDescriptions(disqualified_descriptions, True)

X = qualified_clean_descriptions + disqualified_clean_descriptions
Y = createLabelArray(
    qualified_clean_descriptions, disqualified_clean_descriptions
)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42
)
vectorizer.fit(X_train)
X_train, X_test = transformByVectorizer(X_train, X_test, vectorizer)
tfidf_transformer.fit(X_train)
X_train, X_test = transformByTfIdf(X_train, X_test, tfidf_transformer)


gb_score, gradient = runGradientBoost(X_train, X_test, Y_train, Y_test)

print 'Gradient boosting score: ', gb_score


with open('../event_qualify/algorithms/gradient', 'wb') as f:
    cPickle.dump(gradient, f)



with open('../event_qualify/algorithms/vectorizer', 'wb') as file:
    cPickle.dump(vectorizer, file)

with open('../event_qualify/algorithms/tfidf_transformer', 'wb') as file:
    cPickle.dump(tfidf_transformer, file)
