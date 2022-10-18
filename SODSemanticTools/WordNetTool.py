import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus.reader import Synset
from treelib import Tree
from collections import namedtuple

Node = namedtuple('Node', ['count'])


def get_root_name(input_word):
    input_object = wn.synsets(input_word, pos=wn.NOUN)
    hypernyms_paths = input_object[0].hypernym_paths()
    return hypernyms_paths


if __name__ == '__main__':
    result_list = get_root_name("neck_brace")
    if len(result_list) > 1:
        length = len(result_list[0])
        final_result = result_list[0]
        for result in result_list:
            if len(result) < length:
                final_result = result
    else:
        final_result = result_list
    result_list2 = get_root_name("penis")
    print(final_result[0].name().split('.')[0])
    # print(result_list2)
    # 建立树结构
    tree = Tree()
    for idx, element in enumerate(final_result):
        name = element.name().split('.')[0]
        father_node_id_for_next_node = ''
        if not tree.contains(name):
            if len(father_node_id_for_next_node) == 0:
                tree.create_node(name, name, data=0)
            else:
                tree.create_node()
            father_node_id_for_next_node = name
        else:
            tree.get_node(name).data = tree.get_node(name).data + 1
            father_node_id_for_next_node = name


    tree.merge()
    tree.show(data_property='age')
