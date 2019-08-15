import numpy as np
from collections import Counter
from gensim.models.word2vec import Word2Vec

import json
import random
def sent2vec(model, content):
    words = str(content).lower()
    word_vec = np.zeros((1, 300))

    for w in words:
        if w in model:

            word_vec += np.array([model[w]])
    return word_vec
def load_word2vec(load_path):

    content = []
    word2vec = {}
    with open(load_path, "r", encoding="utf-8") as f:
        LINE = f.readlines()
        for line in LINE:
            id_, en1, en2, sentence = line.strip().split('\t')
            #print("sentence:{}".format(sentence))
            sentence = sentence.split(" ")
            #print("sentence:{}".format(sentence))
            content.append(sentence)
    #print("content:{}".format(content))
    model = Word2Vec(content, sg=1, size=300, min_count=10, window=5, negative=5, sample=1e-4, workers=10)
    model.init_sims(replace=True)
    print(model.wv.index2word)
    for sentence in content:
        for word in sentence:
            if word in model:
                word2vec[word] = model[word]
    #print("word2vec:{}".format(word2vec))
    #js = json.dumps(word2vec)
    #f_txt = open("H:/biebiecompetition/competion/word2vec/word2vec_1.txt", "w", encoding="utf-8")
    #f_txt.write(js)
    #f_txt.close()
    return word2vec
class Read(object):
    def __init__(self, path, label_path_1, label_path_2):
        self.path = path

        self.relation2id_path = label_path_1
        self.label_path = label_path_2
        self.sequence = 60


        self.word2vec = load_word2vec(path)
        self.word_dim = 300
        self.relation2id = {}

        self.pos_limit = int(30)
    def read_word2vec(self):
        path = "H:/biebiecompetition/competion/word2vec/word2vec.txt"
        fr = open(path, "r+", encoding="utf-8")
        word2vec = eval(fr.read())
        print("word2vec:{}".format(word2vec))

    def read_data(self):
        data = []
        with open(self.path, "r", encoding="utf-8") as f:

            LINE = f.readlines()
            for line in LINE[:256]:
                id_, en1, en2, sentence = line.strip().split('\t')

                sentence = sentence.split(" ")
                #print("sentence:{}".format(len(sentence)))
                word2vector = self.content_encoding(sentence)


                p11, p22 = self.pos_encodding(sentence, en1, en2)
                #print("p11:{}".format(max(p11)))
                #print("p22:{}".format(max(p22)))
                #print("p11:{}".format(p11))
                #print("p22:{}".format(p22))
                data.append((word2vector, p11, p22))

            #print("max(data):{}".format(max(data)))
            return data
    def relation_id(self):
        with open(self.relation2id_path, "r", encoding="utf-8") as f:
            LINE = f.readlines()
            for line in LINE:
                relation, id_ = line.strip().split()
                self.relation2id[relation] = id_
        return self.relation2id
    def read_label(self):
        relation2id = self.relation_id()
        self.num_classes = len(relation2id)
        Label = []
        with open(self.label_path, "r", encoding="utf-8") as f:
            LINE = f.readlines()
            for i, line in enumerate(LINE[:1000]):
                #print(i)
                rel = [0] * self.num_classes
                sent_id, label_ = line.strip().split("\t")
                label = int(label_)
                rel[label] = 1
                Label.append(rel)
            print("Label:{}".format(Label))
        return Label

    def content_encoding(self, sentence):
        v = []
        for i, w in enumerate(sentence):
            if w not in self.word2vec:
                tmp = np.zeros(self.word_dim)
            else:
                tmp = self.word2vec[w]
            v.append(tmp)
        #print("v:{}".format(len(v)))
        vectors = self.padding(v)
        return vectors
    def padding(self, vectors):
        a = self.sequence - len(vectors)
        if a > 0:
            front = int(a / 2)
            back = a - front
            front_vec = [np.zeros(self.word_dim) for i in range(front)]
            back_vec = [np.zeros(self.word_dim) for i in range(back)]
            vectors = front_vec + vectors + back_vec
        else:
            vectors = vectors[:self.sequence]

        return vectors

    def pos_encodding(self, sentence, en1, en2):
        l1 = 0
        l2 = 0
        p11 = []
        p22 = []
        for i, w in enumerate(sentence):
            if w == en1:
                l1 = i
            if w == en2:
                l2 = i

        for i, w in enumerate(sentence):
            a = i - l1
            b = i - l2
            if a > self.pos_limit:
                a = self.pos_limit
            if b > self.pos_limit:
                b = self.pos_limit
            if a < -self.pos_limit:
                a = -self.pos_limit
            if b < -self.pos_limit:
                b = -self.pos_limit
            p11.append(a + 31)
            p22.append(b + 31)
        a = self.sequence - len(p11)
        if a > 0:
            front = int(a / 2)
            back = a - front
            front_vec = [0 for i in range(front)]
            back_vec = [0 for i in range(back)]
            p11 = front_vec + p11 + back_vec
            p22 = front_vec + p22 + back_vec
        else:
            p11 = p11[:self.sequence]
            p22 = p22[:self.sequence]
        return p11, p22
    def batcher_yield(self, data_1, label, batch_size):
        content,p1, p2,  labels = [], [], [], []
        for (con, p_1, p_2), l in zip(data_1, label):

            if len(content) == batch_size:
                yield content, p1, p2, labels
                content, p1, p2, labels = [], [], [], []
            content.append(con)
            p1.append(p_1)
            p2.append(p_2)
            labels.append(l)
        if len(content) != 0:
            yield content, p1, p2, labels










if __name__ == "__main__":
    data_path = "H:/biebiecompetition/competion/open_data/sent_train.txt"


    label_path_1 = "H:/biebiecompetition/competion/open_data/relation2id.txt"
    label_path_2 = "H:/biebiecompetition/competion/open_data/sent_relation_train.txt"
    #load_word2vec(data_path)
    read_model = Read(data_path, label_path_1, label_path_2)
    #read_model.read_data()
    data = read_model.read_data()

    #print("data:{}".format(data))
    label = read_model.read_label()
    #print("label:{}".format(label))
    randnum = random.randint(0, 100)
    random.seed(randnum)
    random.shuffle(data)
    random.seed(randnum)
    random.shuffle(label)
    batcher = read_model.batcher_yield(data,label, 2)
    for con, p1, p2, l in batcher:

        print("con:{}".format(con))
        print("p1:{}".format(p1))
        print("p2:{}".format(p2))
        print("l:{}".format(l))




