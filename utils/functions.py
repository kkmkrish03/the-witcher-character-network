import pandas as pd
import numpy as np
import os
import re
import spacy
from spacy import displacy
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import community.community_louvain as community_louvain
from pathlib import Path 
import glob
import psutil


def ner(file_name):
    """Function to process text from a text file (.txt) using Spacy.

    Args:
        file_name (String): name of a txt file as string

    Returns:
        _type_: a processed doc file using Spacy English Language Model
        
    """
    nlp = spacy.load("en_core_web_sm")
    book_text = open(file_name).read()
    book_doc = nlp("../"+book_text)
    return book_doc

def get_ne_list_per_sentence(spacy_doc):
    """Get a list of entities per sentence of a Spacy document and store in a dataframe

    Args:
        spacy_doc (_type_): a Spacy processed document

    Returns:
        dataframe: a dataframe containing the sentences and corresponding list of recognised characters in the sentences
    """
    sentence_entity_df = []
    # Loop through each sentence and store named entity list for each sentence
    for sent in spacy_doc.sents:
        entity_list = [ent.text for ent in sent.ents]
        sentence_entity_df.append({'sentence': sent, 'entities': entity_list})

    sentence_entity_df = pd.DataFrame(sentence_entity_df)
    return sentence_entity_df

def filter_entity(entity_list, character_df):
    """Function to filter out non-character entities eg. ["Geralt", "kkm", "2"] --> ['Geralt']

    Args:
        entity_list (list): list of entities to be filtered
        character_df (dataframe): a dataframe containing characters' names and characters' firstname

    Returns:
        list: a list of entities that are characters (matching by name or first name )
    """
    return [ent for ent in entity_list
            if ent in list(character_df.character)
            or ent in list(character_df.character_firstname)]

def create_relationships(df, window_size):
    """Create a dataframe of relationships based on the df dataframe (containing lists of characters per sentence) and the window size of n sentences

    Args:
        df (dataframe): dataframe containing a column called character_entities with the list of characters for each sentence of a documentation
        window_size (number): size of the windows (number of sentences) for creating relationships between two adjacent characters in the text.

    Returns:
        dataframe: a relationship dataframe containing 3 columns: source, target, value. 
    """
    relationships = []

    for i in range(df.index[-1]):
        end_i = min(i + window_size, df.index[-1])
        char_list = sum((df.loc[i: end_i].character_entities), [])

        # Remove duplicated characters that are next to each other
        char_unique = [char_list[i] for i in range(len(char_list)) 
                       if (i==0) or char_list[i] != char_list[i-1]]

        if len(char_unique) > 1:
            for idx, a in enumerate(char_unique[:-1]):
                b = char_unique[idx + 1]
                relationships.append({"source": a, "target": b})
         
    return pd.DataFrame(relationships)

def weighted_relationship(relationship_df):
    """Sort and sum the number of edges between two nodes to define weight of that relationship

    Args:
        relationship_df (dataframe): dataframe with relationships between characters

    Returns:
        dataframe: dataframe with weighted relationships between characters
    """
    # Convert string value columns to int
    relationship_df['value'] = relationship_df['value'].astype(int)
    # Sort the cases with a->b and b->a
    # relationship_df = pd.DataFrame(np.sort(relationship_df.values, axis = 1), columns = relationship_df.columns)
    relationship_df = relationship_df.groupby(by=["source","target"], sort=False, as_index=False).sum()
                
    return relationship_df

def weighted_relationship_initial(relationship_df):
    """Sort and sum the number of edges between two nodes to define weight of that relationship

    Args:
        relationship_df (dataframe): dataframe with relationships between characters

    Returns:
        dataframe: dataframe with weighted relationships between characters
    """
    # Sort the cases with a->b and b->a
    relationship_df = pd.DataFrame(np.sort(relationship_df.values, axis = 1), columns = relationship_df.columns)
    relationship_df["value"] = 1
    relationship_df = relationship_df.groupby(by=["source","target"], sort=False, as_index=False).sum()
                
    return relationship_df

def merge_dataframes(dfs):
    merged_df = pd.concat(dfs)
    return merged_df

def create_character_relationship_network(GRAPH):
	net = Network(notebook=True, cdn_resources='in_line', width="1000px", height="700px", bgcolor="#222222", font_color="white")
	node_degree = dict(GRAPH.degree)
	#setting up the node size attribute based on degree of the node
	nx.set_node_attributes(GRAPH, node_degree, "size")
	net.from_nx(GRAPH)
	net.show("./graph/witcher_network.html")

def create_degree_centrality_plot(GRAPH, n):
	# Degree Centrality
	degree_dict = nx.degree_centrality(GRAPH)
	degree_df = pd.DataFrame.from_dict(degree_dict, orient='index', columns=['centrality'])
	# Plot Top-10 nodes
	pltDf = degree_df.sort_values('centrality', ascending=False)[0:n-1]
	fig = pltDf.plot(kind="bar").get_figure()
	fig.savefig('./graph/degree_centrality_graph.png')

def create_betweenness_centrality_plot(GRAPH, n):
	# Between-ness centrality
	betweenness_dict = nx.betweenness_centrality(GRAPH)
	betweenness_df = pd.DataFrame.from_dict(betweenness_dict, orient='index', columns=['centrality'])
	# Plot Top-10 nodes
	pltDf = betweenness_df.sort_values('centrality', ascending=False)[0:n-1]
	fig = pltDf.plot(kind="bar").get_figure()
	fig.savefig('./graph/betweenness_centrality_graph.png')

def create_closeness_centrality_plot(GRAPH, n):
	# closeness centrality
	closeness_dict = nx.closeness_centrality(GRAPH)
	closeness_df = pd.DataFrame.from_dict(closeness_dict, orient='index', columns=['centrality'])
	# Plot Top-10 nodes
	pltDf = closeness_df.sort_values('centrality', ascending=False)[0:n-1]
	fig = pltDf.plot(kind="bar").get_figure()
	fig.savefig('./graph/closeness_centrality_graph.png')
 
def create_character_community_detection(GRAPH):
	# Community detection
	communities = community_louvain.best_partition(GRAPH)
	nx.set_node_attributes(GRAPH, communities, "group")
	community_net = Network(notebook=True, cdn_resources='in_line', width="1000px", height="700px", bgcolor="#222222", font_color="white")
	community_net.from_nx(GRAPH)
	community_net.show("./graph/witcher_communities.html")

def compute_evolution_of_main_five_characters_importance(books_graph):
	# Evolution of Characters importance
	# Creating a list of degree centrality of all the books
	evol = [nx.degree_centrality(book) for book in books_graph]
	# Creating a DatFrame from the list of degree centralities in all the books
	degree_evol_df = pd.DataFrame.from_records(evol)
	# Plotting the degree centrality evolution of 5 main characters
	pltDf = degree_evol_df[["Geralt", "Ciri", "Yennefer", "Dandelion", "Vesemir"]]
	fig = pltDf.plot().get_figure()
	fig.savefig('./graph/degree_centrality_evolution_graph.png')
 
def plot_kamada_kawai_layout_graph(GRAPH):
    pos = nx.kamada_kawai_layout(GRAPH)
    nx.draw(GRAPH, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos = pos)
    plt.show()
    plt.savefig('./graph/kamada_kawai_layout_graph.png')
    
def get_witcher_character_list():
    character_df = pd.read_csv('./data/characters.csv')
    # Remove unwanted texts and do cleaning of character names
    character_df['character'] = character_df['character'].apply(lambda x: re.sub("[\(].*?[\)]", "", x))
    character_df['character_firstname'] = character_df['character'].apply(lambda s: s 
                                                                            if s.startswith("The ") or 
                                                                            s.startswith("A ") or
                                                                            s.startswith("White ") or
                                                                            s.startswith("An ") 
                                                                            else s.split(' ', 1)[0])
    return character_df

def get_all_witcher_books():
    all_books = [b for b in os.scandir('./data/books') if '.txt' in b.name]
    #Sort dir entries by name
    all_books.sort(key=lambda x: x.name)
    return all_books

def do_sentence_character_mapping(book, character_df):
    book_text = ner(book)
        
    # Get list of entities per sentences
    sentence_entity_df = get_ne_list_per_sentence(book_text)
        
    # select only character entities
    sentence_entity_df['character_entities'] = sentence_entity_df['entities'].apply(lambda x: filter_entity(x, character_df))

    # Filter out sentences that don't have have any character entities
    sentence_entity_df_filtered = sentence_entity_df[sentence_entity_df['character_entities'].map(len) > 0]
        
    # Take only the first name of character
    sentence_entity_df_filtered['character_entities'] = sentence_entity_df_filtered['character_entities'].apply(lambda item: [s if s.startswith("The ") or s.startswith("A ") or s.startswith("An ") or s.startswith("White ") else s.split(' ', 1)[0] for s in item])
    
    return sentence_entity_df_filtered
 
 
def gather_files():
    # Use glob to get a list of all CSV files in a directory
    csv_files = glob.glob('./data/temp/*.csv')
    # Initialize an empty list to store individual DataFrames
    data_frames = []
    # Loop through each CSV file and read it into a DataFrame
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        data_frames.append(df)
    return data_frames

def get_status_report():
    # Get memory usage
    memory_usage = psutil.virtual_memory()
    print(f"Total Memory: {memory_usage.total / (1024 ** 3):.2f} GB")
    print(f"Available Memory: {memory_usage.available / (1024 ** 3):.2f} GB")
    print(f"Used Memory: {memory_usage.used / (1024 ** 3):.2f} GB")

    # Get CPU usage
    cpu_usage = psutil.cpu_percent(interval=1)
    print(f"CPU Usage: {cpu_usage}%")
    

def clean_temp(directory_path):
    # Get a list of all files in the directory
    file_list = os.listdir(directory_path)

    # Loop through the list and delete each file
    for file_name in file_list:
        file_path = os.path.join(directory_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted {file_name}")

    print("All files have been deleted.")
