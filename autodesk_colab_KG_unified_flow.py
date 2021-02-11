#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 12:10:13 2020

@author: vinny
"""

### Information ###
# This sample code is designed to create a knowledge graph from data supplied from the OSDR
# The data collected includes: Component, System, Functon, Flow, and Material Data
# The edges are denoted by Flow in, Flow out, and Familial Assembly.
# This is a rough code, not following best practices


import networkx as nx
import csv
import numpy as np


def load_data(path='./data/autodesk_colab_fullv3_202010291746.csv'):
    with open(path, 'rt') as f:
        reader = csv.reader(f)
        data = list(reader)


    ## list of Systems within the dataset
    systems = [item[6] for item in data]
    unique_systems = np.sort(list(set(systems)))
    ## Droping "systems" label from CSV as a unique value ##
    unique_systems = unique_systems[0:len(unique_systems)-1]





    ####################### LABEL GRAPH for visualization ##############################
    ## Adding Nodes from CSV all component-function combonations based on Common name component Labels ##



    ## List of NetworkX Graphs##
    graph_list = []

    for h in range(len(unique_systems)):


    ## Creating System_level Sub Dataset   ##
        system_data = []
        for q in range(len(data)):
            if unique_systems[h] == data[q][6]:
                system_data.append(data[q])

    ## Intializing Graph ##
        G = nx.DiGraph()

    #################### NODE CREATION ###################################
    ## Adding node and attributes for all data ##
        for i in range(len(system_data)):


            #Generate Node
            G.add_node((system_data[i][0],system_data[i][2]))

            ## Adding Attributes to each node

            ## Component ID ##
            G.nodes[(system_data[i][0],system_data[i][2])][data[0][1]] = system_data[i][1]

            ## Component Basis Name ##
            G.nodes[(system_data[i][0],system_data[i][2])][data[0][5]] = system_data[i][5]

            ## Product Name ##
            G.nodes[(system_data[i][0],system_data[i][2])][data[0][11]] = system_data[i][11]

            ## Product Basis Name##
            G.nodes[(system_data[i][0],system_data[i][2])][data[0][13]] = system_data[i][13]

            ## Material ##
            G.nodes[(system_data[i][0],system_data[i][2])][data[0][15]] = system_data[i][15]

            ## Function ##
            G.nodes[(system_data[i][0],system_data[i][2])][data[0][9]] = system_data[i][9]


            #####  Hierarchical Functions ###############

            ## Component Function is natively tier 1 ####
            if system_data[i][24] == str(1):

                ## Function Tier 1##
                G.nodes[(system_data[i][0],system_data[i][2])]['tier_1_function'] = system_data[i][9]

                ## Function Tier 2##
                G.nodes[(system_data[i][0],system_data[i][2])]['tier_2_function'] = ''

                ## Function Tier 3##
                G.nodes[(system_data[i][0],system_data[i][2])]['tier_3_function'] = ''

            ## Component Function is natively tier 2 ####
            elif system_data[i][24] == str(2):

                ## Function Tier 1##
                G.nodes[(system_data[i][0],system_data[i][2])]['tier_1_function'] = system_data[i][27]

                ## Function Tier 2##
                G.nodes[(system_data[i][0],system_data[i][2])]['tier_2_function'] = system_data[i][9]

                ## Function Tier 3##
                G.nodes[(system_data[i][0],system_data[i][2])]['tier_3_function'] = ''

            ## Component Function is natively tier 3 ####
            elif system_data[i][24] == str(3):

                ## Function Tier 1##
                G.nodes[(system_data[i][0],system_data[i][2])]['tier_1_function'] = system_data[i][30]

                ## Function Tier 2##
                G.nodes[(system_data[i][0],system_data[i][2])]['tier_2_function'] = system_data[i][27]

                ## Function Tier 3##
                G.nodes[(system_data[i][0],system_data[i][2])]['tier_3_function'] = system_data[i][9]


            ## Component Function does not exist ####
            else:

                ## Function Tier 1##
                G.nodes[(system_data[i][0],system_data[i][2])]['tier_1_function'] = ''

                ## Function Tier 2##
                G.nodes[(system_data[i][0],system_data[i][2])]['tier_2_function'] = ''

                ## Function Tier 3##
                G.nodes[(system_data[i][0],system_data[i][2])]['tier_3_function'] = ''




    #################### EDGE CREATION ###################################

        ## Adding Edges first by Assembly Data ##


        # for the length of data
        # point 1 take id
        # Pull G node index
        # take child off
        # search data for child of id == id
        # Create G node Index
        # Creat edge ftom gnode index 1 to 2


        node_index = list(G.nodes)
        for k in range(len(G.nodes.data())):
            data_point_1 = node_index[k][0]
            child_of = data[int(data_point_1)][3]

            for j in range(len(G.nodes.data())):
                if child_of == data[int(node_index[j][0])][1]:
                    G.add_edge(node_index[k], node_index[j])
                    G.add_edge(node_index[j], node_index[k])

        # Adding Edges by Flow Data ##

        # ##  Input Flow ##
        for l in range(len(G.nodes.data())):
            data_point_1 = node_index[l][0]
            input_flow_from = data[int(data_point_1)][19]

            for m in range(len(G.nodes.data())):
                if input_flow_from  == data[int(node_index[m][0])][1]:
                    G.add_edge(node_index[m],node_index[l])
                    G.edges[node_index[m],node_index[l]]['flow'] = data[int(data_point_1)][17]


        # ## Output Flow ##
        for n in range(len(G.nodes.data())):
            data_point_1 = node_index[n][0]
            output_flow_from = data[int(data_point_1)][23]

            for p in range(len(G.nodes.data())):
    
                if output_flow_from  == data[int(node_index[p][0])][1]:
                    G.add_edge(node_index[n], node_index[p])
                    
                    ## Checking if flows already exist on for the edge
                    if any(G.edges[node_index[n], node_index[p]].values()) == True:
                        
                        ## checking if the input/out are matching, if not the new flow is added to the edge
                        if G.edges[node_index[n], node_index[p]]['flow'] != data[int(data_point_1)][21]:
                            G.edges[node_index[n], node_index[p]]['flow'] = G.edges[node_index[n], node_index[p]]['flow'],data[int(data_point_1)][21]
                    
                    else:
                        
                        G.edges[node_index[n], node_index[p]]['flow'] = data[int(data_point_1)][21]



        ## Adding System Knowledge Graph to Graph List #########
        graph_list.append(G)



    nx.draw(graph_list[0],with_labels = True)
    # print(len(unique_systems))
    return graph_list


if __name__ == '__main__':
    graphs = load_data()
    