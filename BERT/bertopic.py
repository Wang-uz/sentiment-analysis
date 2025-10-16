
# pre-processing

import pandas as pd
import os
from itertools import chain

isnull = lambda x: True if len(str(x).strip()) == 0 \
                           or (str(x).lower() in ['nan', 'none', r'\N', r'\n']) \
    else False

notnull = lambda x: False if isnull(x) else True
extend_list = lambda x: list(chain(*x))


def cut_sent(para):
    import re
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)
    para = re.sub('([，,!;；。！？\?])([^”’])', r"\1\n\2", para)
    para = re.sub('([，,!;；。！？\?][”’])([^，,;；!，。！？\?])', r'\1\n\2', para)
    para = para.rstrip()
    return para.split("\n")


def read_txt(path, line_split=True):
    try:
        with open(path, encoding='utf-8') as f:
            l = [x.strip() for x in f.readlines() if len(x.strip()) != 0]
    except:
        with open(path, encoding='gbk') as f:
            l = [x.strip() for x in f.readlines() if len(x.strip()) != 0]

    if line_split:
        return l
    else:
        return [' '.join(l)]


def have_chinese(ss):
    for s in ss:
        if s >= u'\u4e00' and s <= u'\u9fa5':
            return True
    return False


def all_chinese(ss):
    for s in ss:
        if not ((s >= u'\u4e00' and s <= u'\u9fa5')):
            return False
    return True


def word_filter(s):
    if '一' == s[0]:
        return False

    if '二' == s[0]:
        return False

    if '三' == s[0]:
        return False

    if '四' == s[0]:
        return False

    if len(s) < 2:
        return False
    if not all_chinese(s.replace(' ', '')):
        return False
    return True


def filter_text(s):
    s = str(s)
    if len(s.strip()) < 3:
        return False
    if not have_chinese(s.strip()):
        return False
    return True


OUTPUT_DIR = r"results"
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

stopwords = set(read_txt(r"stopwords.txt"))
setword_mine = list(set((read_txt(r"setwords.txt"))))

import jieba

for w in setword_mine:
    jieba.add_word(w)

tokenizer = lambda x: [xx for xx in jieba.cut(str(x).lower()) if
                       len(xx.strip()) > 0 and (xx.strip() not in stopwords) and word_filter(xx.strip())]


def DBC2SBC(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 0x3000:
            inside_code = 0x0020
        else:
            inside_code -= 0xfee0
        if not (0x0020 <= inside_code and inside_code <= 0x7e):
            rstring += uchar
            continue
        rstring += chr(inside_code)
    return rstring

import os
import pandas as pd

df = pd.read_excel(r'data\全部数据的情感预测结果.xlsx')

df['text'] = df.apply(lambda x: x['sentence'], axis=1)
df


### topic analysis


name = ''
d1 = df.__deepcopy__()
list_text = list(set(d1['sentence']))

len(list_text)


from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired
from bertopic.representation import MaximalMarginalRelevance
from transformers.pipelines import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

n_topics = 20

embedding_model = SentenceTransformer(r"paraphrase-multilingual-MiniLM-L12-v2")

dim_model = UMAP(n_components=10, random_state=20)

cluster_model = KMeans(n_clusters=n_topics)

vectorizer_model = CountVectorizer(analyzer='word',
                                   tokenizer=tokenizer, stop_words=stopwords, ngram_range=(1, 1))
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

representation_model = MaximalMarginalRelevance(diversity=0.3)

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=dim_model,
    hdbscan_model=cluster_model,
    vectorizer_model=vectorizer_model,
    ctfidf_model=ctfidf_model,
    representation_model=representation_model,
    nr_topics=n_topics
)
topics, probs = topic_model.fit_transform(list_text)

print(topic_model.get_topic_info())

# %%

dd = {i: "Topic " + str(i + 1) for i in range(n_topics)}

topic_model.set_topic_labels(dd)
fig = topic_model.visualize_barchart(top_n_topics=len(set(topics)), n_words=10, custom_labels=True,
                                     width=300, height=300)
fig.write_html(os.path.join(OUTPUT_DIR, name + "file1.html"))

ll = topic_model.generate_topic_labels(nr_words=3, topic_prefix=False,
                                       word_length=10,
                                       separator="_")
dd = {i: "Topic_" + str(i + 1) + '_' + ll[i] for i in range(n_topics)}

topic_model.set_topic_labels(dd)

print(topic_model.get_topic_info())
fig

# %%

dict_topic = dict(zip(list_text, [x for x in topics]))

d1['topic_bertopic'] = d1.apply(lambda x: dict_topic[x['sentence']] + 1, axis=1)

d1.to_excel(os.path.join(OUTPUT_DIR, name + '_topic_analysis_bertopic.xlsx'), index=False)

df_res = topic_model.get_topic_info()
df_res['keywords_c-tfidf_score'] = df_res.apply(lambda x: topic_model.get_topic(x['Topic']), axis=1)
df_res.to_excel(os.path.join(OUTPUT_DIR, name + '_topic_analysis_bertopic_information.xlsx'), index=False)

dd = pd.DataFrame(df_res, columns=['Topic', 'keywords_c-tfidf_score'])
dd.columns = ['topic', 'terms']
dd = dd.explode('terms')
dd['word'] = dd.apply(lambda x: x['terms'][0], axis=1)
dd['score'] = dd.apply(lambda x: x['terms'][1], axis=1)
dd['topic'] = dd.apply(lambda x: 'topic_' + str(x['topic'] + 1), axis=1)
dd.drop("terms", axis=1, inplace=True)
dd.to_excel(os.path.join(OUTPUT_DIR, name + '_keywords_bertopic.xlsx'), index=False)
print(dd.head(5))

fig = topic_model.visualize_documents(list_text, width=1200, height=1000, custom_labels=True)
fig.write_html(os.path.join(OUTPUT_DIR, name + "file2.html"))

fig = topic_model.visualize_heatmap(width=1200, height=900, custom_labels=True)
fig.write_html(os.path.join(OUTPUT_DIR, name + "file3.html"))

fig = topic_model.visualize_hierarchy(width=1200, height=1000, custom_labels=True)
fig.write_html(os.path.join(OUTPUT_DIR, name + "file4.html"))

fig = topic_model.visualize_topics(width=1000, height=1000, custom_labels=True)
fig.write_html(os.path.join(OUTPUT_DIR, name + "file5.html"))

fig = topic_model.visualize_topics(width=1000, height=1000, custom_labels=True)
fig.write_html(os.path.join(OUTPUT_DIR, name + "file5.html"))
fig

