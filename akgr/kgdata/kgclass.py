from random import sample, choice, randint
import numpy as np

import networkx as nx

from tqdm import tqdm

# The class that is used for sampling from a networkx graph.
class GraphSampler:
    def __init__(self, graph, id2rel):

        self.graph = graph
        # self.re_graph = graph.reverse(copy=True)
        self.dense_nodes = list(self.graph.nodes)
        self.id2rel = id2rel
        self.out_degree_data = self.preprocess_out_degree_by_key()

    def in_degree(self, nodes):
        """
        https://networkx.org/documentation/stable/reference/classes/generated/networkx.MultiDiGraph.in_degree.html#networkx.MultiDiGraph.in_degree
        """
        return self.graph.in_degree(nodes)
    def in_edges(self, node: int):
        """
        https://networkx.org/documentation/stable/reference/classes/generated/networkx.MultiDiGraph.in_edges.html
        """
        # list of (u, v, k)
        if self.graph.has_node(node):
            return self.graph.in_edges(node, keys=True)
        else:
            return []

    def preprocess_out_degree_by_key(self):
        out_degree_data = {}
        for u in self.graph.nodes:
            out_degree_data[u] = {}
            out_edges = self.out_edges(u)
            for _, v, k in out_edges:
                # initialize
                if not k in out_degree_data[u]: out_degree_data[u][k] = 0
                # count
                out_degree_data[u][k] += 1
        return out_degree_data


    def out_degree_by_key(self, node, key):
        return self.out_degree_data[node][key]

    def out_edges_by_key(self, node, key):
        out_edges = self.out_edges(node)
        out_edges_key = []
        for u, v, k in out_edges:
            if k == key: out_edges_key.append((u, v, k))
        return out_edges_key

    def out_edges(self, nodes):
        """
        https://networkx.org/documentation/stable/reference/classes/generated/networkx.MultiDiGraph.out_edges.html
        """
        # list of (u, v, k)
        return self.graph.out_edges(nodes, keys=True)

    def is_reverse_edge(self, id1, id2):
        """
        Input: id1, id2 the indices (=keys) of two edges.
        """
        if id1 == None or id2 == None:
            return False
        rel1 = self.id2rel[int(id1)]
        rel2 = self.id2rel[int(id2)]
        # print(rel1, rel2)
        return (rel1[1:] == rel2[1:]) and (set([rel1[0], rel2[0]]) == set(['+', '-']))

    def iterative_sample_with_pattern(self, pattern="(p,(e))"):

        result_query_list = []
        for node in tqdm(self.dense_nodes):
            _query, _, _ = self.recur_sample_query_given_pattern_answer(pattern, node)
            if _query is not None:
                result_query_list.append(_query)

        return result_query_list

    def generate_one_p_queries(self):
        result_query_list = []
        for node in tqdm(self.dense_nodes):
            for tail_node, attribute_dict in self.graph[node].items():
                # "(p,(40),(e,(2429)))"
                for key in attribute_dict.keys():
                    result_query_list.append("(p,(" + str(key) + "),(e,(" + str(node) + ")))")

        return list(set(result_query_list))

    # The function used to call the recursion of sampling queries from the ASER graph.
    def sample_valid_query_given_pattern(self, pattern):
        while True:
            tail_node = sample(self.dense_nodes, 1)[0] # randomly select an answer node
            _query, _, _ = self.recur_sample_query_given_pattern_answer(pattern, tail_node)
            if _query is not None:
                return _query


    def extract_operator_subqueries(self, pattern):
        """
        Input:
        - pattern (query): of the form "(operator, operand_1, ..., operand_n)"

        Output:
        - operator
        - sub_queries: [operator, operand_1, ..., operand_n]
        """
        pattern = pattern[1:-1] # remove the outmost parenthesis
        parenthesis_count = 0

        operator = pattern[0]
        sub_queries = [operator]
        # j = 0
        l = 0; r = 0

        for i, element in enumerate(pattern):
            # Skip the comma inside a parenthesis
            if element == "(":
                parenthesis_count += 1
                if parenthesis_count == 1:
                    l = i

            elif element == ")":
                parenthesis_count -= 1
                if parenthesis_count == 0:
                    r = i
                    sub_queries.append(pattern[l: r+1])


            # if parenthesis_count > 0:
            #     continue

            # if element == ",":
                # sub_queries.append(pattern[j: i])
                # j = i + 1

        # sub_queries.append(pattern[j: len(pattern)])
        # print(pattern)
        # print(sub_queries)
        # operator = sub_queries[0]

        return operator, sub_queries

    def recur_sample_query_given_pattern_answer(self, pattern: str, tail_node):
        """
        Input:
        - pattern: "(operator, [edge], operand_1, ..., operand_n)",
            operator([edge], operand_1, ..., operand_n) = answer node.
            form 1: (p, (edge), operand_1);
            form 2: (i/u, operand_1, ..., operand_n);
            form 3: (n, operand_1);
            operands are sub queries (or anchor nodes)
        - tail_node

        Output:
        - str: the sampled query string
        - node (optional):
            - the only use of it is to ensure the previous nodes of sub queries
              are distinct when the operator="i"
            - so for predecessor operators of "i", if in that level is operator(prev, answer),
              then return head_node instead of tail_node
        """
        operator, sub_pattern = self.extract_operator_subqueries(pattern)

        # projection has only one operand
        if operator == "p":
            if self.in_degree(tail_node) == 0:
                return None, None, None

            head_node, _, relation = choice(list(self.in_edges(tail_node)))

            sub_query, _, prev_relation = self.recur_sample_query_given_pattern_answer(sub_pattern[1], head_node)
            if sub_query is None:
                return None, None, None
            if self.is_reverse_edge(prev_relation, relation):
                return None, None, None

            # return f'(p,({relation}),{sub_query})', head_node, relation
            # represent relation as negative number
            return ['(', 'p', '(', -relation, ')', *sub_query, ')'], head_node, relation

        elif operator == "n":
            """If we use the negation here, it is possible that we generate a query that do not have an answer.
            But the overall chance is small. Anyway, when we cannot find an answer we just sample again.

            After modification, we choose to use the same node for sampling to enable the negation query do have an effect
            on the final outcome
            """

            sub_query, head_node, relation = self.recur_sample_query_given_pattern_answer(sub_pattern[1], tail_node)
            if sub_query is None:
                return None, None, None

            # return ?
            # return f'(n,{sub_query})', head_node, relation
            return ['(', 'n', *sub_query, ')'], head_node, relation

        elif operator == "e":
            # return f'(e,({tail_node}))', None, None
            return ['(', 'e', '(', tail_node, ')', ')'], None, None

        elif operator == "i":

            sub_queries_list = []
            from_node_list = []

            for pattern in sub_pattern[1:]:
                sub_q, head_node, relation = self.recur_sample_query_given_pattern_answer(pattern, tail_node)

                if (sub_q is None) or (sub_q in sub_queries_list) or (head_node in from_node_list):
                    return None, None, None
                # print(f'{head_node}({sub_q}) -> {tail_node}')
                sub_queries_list.append(sub_q)
                from_node_list.append(head_node)

            # print(sub_queries_list)
            # return_str = "(i"
            return_list = ['(', 'i']
            for sub_query in sub_queries_list:
                # return_str += ","
                # return_str += sub_query
                return_list.extend(sub_query)
            # return_str += ")"
            return_list.append(')')

            # return return_str, tail_node, None
            return return_list, tail_node, None

        elif operator == "u":
            # randomly sample a node
            sub_queries_list = []

            # The answer only need to be one of the answers of all sub queries
            random_subquery_index = randint(1, len(sub_pattern) - 1)

            head_node = None
            for i in range(1, len(sub_pattern)):
                if i == random_subquery_index:
                    sub_q, head_node, relation = self.recur_sample_query_given_pattern_answer(sub_pattern[i], tail_node)
                else:
                    sub_q, _, relation = self.recur_sample_query_given_pattern_answer(sub_pattern[i], sample(list(self.graph.nodes()), 1)[0])

                if sub_q is None:
                    return None, None, None

                sub_queries_list.append(sub_q)

            if len(sub_queries_list) == 0:
                return None, None, None

            return_str = "(u"
            return_list = ['(', 'u']
            for sub_query in sub_queries_list:
                # return_str += ","
                # return_str += sub_query
                return_list.extend(sub_query)
            # return_str += ")"
            return_list.append(')')

            # original: return tail_node
            # now: return head_node
            # return return_str, tail_node, None
            return return_list, tail_node, None
        else:
            print("Invalid Pattern in generating query")
            print("Operator: ", operator)


    # The function used for finding the answers to a query
    def search_answers_to_query(self, query: list):

        operator, sub_queries = self.extract_operator_subqueries(query)

        if operator == "p":

            sub_query_answers = self.search_answers_to_query(sub_queries[2])
            # relation_name = int(sub_queries[1][1:-1])
            relation_name = -sub_queries[1][1]
            if sub_queries[1][1] > 0:
                print("# Error: relation > 0")
                exit()
            all_answers = []
            for u, v, k in self.out_edges(sub_query_answers):
                if k == relation_name:
                    all_answers.append(v)
            all_answers = list(set(all_answers))
            return all_answers

        elif operator == "e":
            # return list(set([int(sub_queries[1][1:-1])]))
            return list(set([sub_queries[1][1]]))

        elif operator == "i":

            sub_query_answers_list = []

            for i in range(1, len(sub_queries)):
                sub_query_answers_i = self.search_answers_to_query(sub_queries[i])
                sub_query_answers_list.append(sub_query_answers_i)

            merged_answers = set(sub_query_answers_list[0])
            for sub_query_answers in sub_query_answers_list:
                merged_answers = merged_answers & set(sub_query_answers)

            merged_answers = list(set(merged_answers))

            return merged_answers

        elif operator == "u": # similar to "i"

            sub_query_answers_list = []
            for i in range(1, len(sub_queries)):
                sub_query_answers_i = self.search_answers_to_query(sub_queries[i])
                sub_query_answers_list.append(sub_query_answers_i)

            merged_answers = set(sub_query_answers_list[0])
            for sub_query_answers in sub_query_answers_list:
                merged_answers = merged_answers | set(sub_query_answers)

            merged_answers = list(set(merged_answers))

            return merged_answers

        elif operator == "n":
            sub_query_answers = self.search_answers_to_query(sub_queries[1])
            all_nodes = list(self.graph.nodes)
            negative_answers = [node for node in all_nodes if node not in sub_query_answers]

            negative_answers = list(set(negative_answers))
            return negative_answers

        else:
            print("Invalid Pattern in finding answers")
            print("Operator: ", operator)


    # The function used for finding a query that have at least one answer
    # def sample_valid_question_with_answers(self, pattern):
    #     while True:
    #         _query = self.sample_with_pattern(pattern)
    #         _answers = self.search_answers_to_query(_query)
    #         if len(_answers) > 0:
    #             return _query, _answers
class KG:
    def __init__(self, num_ent, num_rel, ent_id2name, rel_id2name, rel_id2inv,
                 graphs):
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.ent_id2name = ent_id2name
        self.rel_id2name = rel_id2name
        self.rel_id2inv = rel_id2inv
        self.graph_samplers = {
            'train': GraphSampler(graphs['train'], rel_id2name),
            'valid': GraphSampler(graphs['valid'], rel_id2name),
            'test': GraphSampler(graphs['test'], rel_id2name)
        }
        self.num_train_edges = len(self.graph_samplers['train'].graph.edges())