import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus.reader import Synset
from treelib import Tree
from collections import namedtuple


class TreeNodeData:
    def __init__(self, count,files_path_list):
        self.count = count,
        self.files_path_list = files_path_list


def get_root_name(input_word):
    input_object = wn.synsets(input_word, pos=wn.NOUN)
    hypernyms_paths = input_object[0].hypernym_paths()
    return hypernyms_paths


def build_tree(tree, paths_result_list, file_path):
    # 选择最短路径
    if len(paths_result_list) > 1:
        length = len(paths_result_list[0])
        final_result = paths_result_list[0]
        for result in paths_result_list:
            if len(result) < length:
                final_result = result
    else:
        final_result = paths_result_list[0]
    # 父节点名称
    father_node_id_for_next_node = ''
    for idx, element in enumerate(final_result):
        name = element.name().split('.')[0]
        if not tree.contains(name):
            # 为根节点
            tree_node_data = TreeNodeData(count=int(1), files_path_list=[file_path])
            if len(father_node_id_for_next_node) == 0 and idx == 0:
                tree.create_node(tag=name, identifier=name, data=tree_node_data)
                father_node_id_for_next_node = name
            else:
                tree.create_node(tag=name, identifier=name, parent=father_node_id_for_next_node,
                                 data=tree_node_data)
                father_node_id_for_next_node = name
        else:
            node_data = tree.get_node(nid=name).data
            tree_node_data = TreeNodeData(count=node_data.count[0] + 1, files_path_list=node_data.files_path_list.append(file_path))
            tree.update_node(nid=name, data=tree_node_data)
            father_node_id_for_next_node = name
    tree.show()


if __name__ == '__main__':
    result_list2 = get_root_name("penis")
    path_result_list = get_root_name("neck_brace")
    tree = Tree()
    build_tree(tree, path_result_list, '../123455')
    build_tree(tree, result_list2, '../123455')
