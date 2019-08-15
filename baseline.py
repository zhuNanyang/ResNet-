import tensorflow as tf
import numpy as np
from collections import Counter






def read_data(path, word_map):
    sentence_dict = {}
    with open(path, "r", encoding="utf-8") as f:
        LINE = f.readlines()

        for line in LINE[:100]:
            id_, en1, en2, sentence = line.strip().split('\t')
            #print("id_:{}".format(id_))
            #print("en1:{}".format(en1))
            #print("en2:{}".format(en2))
            #print("sentence:{}".format(sentence))
            sentence = sentence.split()
            #print("sentence:{}".format(sentence))
            en1_pos = 0
            en2_pos = 0
            for i in range(len(sentence)):
                if sentence[i] == en1:
                    en1_pos = i
                if sentence[i] == en2:
                    en2_pos = i
            #print("en1_pos:{}".format(en1_pos))
            #print("en2_pos:{}".format(en2_pos))
            words = []
            pos1 = []
            pos2 = []
            length = min(60, len(sentence))
            #print("length:{}".format(length))
            for i in range(length):
                words.append(word_map.get(sentence[i],word_map['UNK']))
                #print("words:{}".format(words))
                pos1.append(pos_index(i - en1_pos))
                pos2.append(pos_index(i - en2_pos))
            print("pos1:{}".format(len(pos1)))
            print("pos2:{}".format(pos2))
            if length < 60:
                for i in range(length, 60):
                    words.append(word_map['PAD'])
                    #print("words:{}".format(words))
                    pos1.append(pos_index(i - en1_pos))
                    pos2.append(pos_index(i - en2_pos))
            print("pos1:{}".format(len(pos1)))
            print("pos2:{}".format(pos2))
            sentence_dict[id_] = np.reshape(np.asarray([words, pos1, pos2], dtype=np.int32), (1, 3, 60))
            #print("np.asarray([words, pos1, pos2]):{}".format(np.asarray([words, pos1, pos2]).shape))
            #print("sentence_dict:{}".format(sentence_dict))
    return sentence_dict










def pos_index(x):
    if x < -15:
        return 0
    if x >= -15 and x <= 15:
        return x + 15 + 1
    if x > 15:
        return 2 * 15 + 2
def load_wordMap(data_path):
    wordMap = {}
    wordMap['PAD'] = len(wordMap)
    wordMap['UNK'] = len(wordMap)
    all_content = []
    for line in open(data_path, encoding="utf-8"):
        #print("line:{}".format(line))
        all_content += line.strip().split('\t')[3].split()
    for item in Counter(all_content).most_common():
        #print("item:{}".format(item))
        if item[1] > 5:
            wordMap[item[0]] = len(wordMap)
        else:
            break
    #print("wordMap:{}".format(wordMap))

    return wordMap
def relation_id():
    relation2id = {}
    path = "H:/biebiecompetition/competion/open_data/relation2id.txt"
    with open(path, "r", encoding="utf-8") as f:
        LINE = f.readlines()
        for line in LINE:
            relation, id_ = line.strip().split()
            relation2id[relation] = id_
        return relation2id
def data_batcher(sentence_dict, filename, padding=False, shuffle=True):
    all_sent_ids = []
    all_sents = []
    all_labels = []
    relation2id = relation_id()
    num_classes = len(relation2id)
    #print("relation2id:{}".format(relation2id))
    #print("num_classes:{}".format(num_classes))
    with open(filename, "r", encoding="utf-8") as f:
        LINE = f.readlines()
        for line in LINE[:2]:
            rel = [0] * num_classes
            sent_id, types = line.strip().split('\t')
            type_list = types.split()
            #print("type_list:{}".format(len(type_list)))
            for tp in type_list:
                print(tp)
                if len(type_list) > 1 and tp == '0':  # if a sentence has multiple relations, we only consider non-NA relations
                    continue
                rel[int(tp)] = 1
            #print("rel:{}".format(rel))
            all_sent_ids.append(sent_id)
            all_sents.append(sentence_dict[sent_id])
            all_labels.append(np.reshape(np.asarray(rel, dtype=np.float32), (-1, num_classes)))
        #print("all_labels:{}".format(all_labels))
        #print("all_labels:{}".format(np.array(all_labels).shape))
        #print("all_sent_ids:{}".format(all_sent_ids))
        #print("all_sents:{}".format(np.array(all_sents).shape))
        data_size = len(all_sent_ids)
        datas = all_sent_ids
        all_sents = np.concatenate(all_sents, axis=0)
        all_labels = np.concatenate(all_labels, axis=0) # np.concatenate(list, axis=0) 相当于将列表里的数组进行连接
        #print("all_sents:{}".format(all_sents.shape))
        #print("all_labels:{}".format(all_labels))
        #print("all_labels.shape:{}".format(all_labels.shape))
        #print("all_sents:{}".format(all_sents))
        data_order = list(range(data_size))
        #print("all_sents[0]:{}".format(all_sents[0]))
        """
        if shuffle:
            np.random.shuffle(data_order)
        for i in range(len(data_order) // 50):
            idx = data_order[i * 50:(i + 1) * 50]
            yield all_sents[idx], all_labels[idx], None
        """


if __name__ == "__main__":
    data_path = "H:/biebiecompetition/competion/open_data/sent_train.txt"

    sent_path = "H:/biebiecompetition/competion/open_data/sent_relation_train.txt"
    word_map = load_wordMap(data_path)
    sent_train = read_data(data_path, word_map)
    data_batcher(sent_train, sent_path, padding=False, shuffle=True)
