import pandas as pd
import numpy as np
import spacy
from spacy import displacy
import networkx as nx
import matplotlib.pyplot as plt
import os
import re
from pyvis.network import Network
import community.community_louvain as community_louvain
from utils.functions import *
import psutil



def start_plotting(books_graph):
    
    # Plot and save evolution of importance of five main characters over the course of books
    compute_evolution_of_main_five_characters_importance(books_graph)
    print("# Plot and save evolution of importance of five main characters over the course of books")
    get_status_report()
    
    # This removes the reference to the object and may free up its memory.
    del books_graph  
    print("# This removes the reference to the object and may free up its memory.")
    get_status_report()
    
    # Read all the temporary stored dataframe
    all_books_relationship = gather_files()
    print("# Read all the temporary stored dataframe")
    get_status_report()
    
    # Merge the list of dataframes to one
    complete_character_relationship = merge_dataframes(all_books_relationship)
    print("# Merge the list of dataframes to one")
    get_status_report()
    
    # Aggregate complete weight of all the relations based on all the books
    weighted_complete_character_relationship = weighted_relationship(complete_character_relationship)
    print("# Aggregate complete weight of all the relations based on all the books")
    get_status_report()
    
    # Create a centralized graph from a pandas dataframe
    GRAPH = nx.from_pandas_edgelist(weighted_complete_character_relationship, 
                                    source = "source",
                                    target = "target",
                                    edge_attr  = "value",
                                    create_using = nx.Graph())
    print("# Create a centralized graph from a pandas dataframe")
    get_status_report()
    
    # Plot and save kamada kawai layout graph
    plot_kamada_kawai_layout_graph(GRAPH)
    print("# Plot and save kamada kawai layout graph")
    get_status_report()
    
    # Plot and save character relationship network graph
    create_character_relationship_network(GRAPH)
    print("# Plot and save character relationship network graph")
    get_status_report()
    
    # Plot and save degree centrality graph of top 10 characters
    create_degree_centrality_plot(GRAPH, 10)
    print("# Plot and save degree centrality graph of top 10 characters")
    get_status_report()
    
    # Plot and save betweenness centrality graph of top 10 characters
    create_betweenness_centrality_plot(GRAPH, 10)
    print("# Plot and save betweenness centrality graph of top 10 characters")
    get_status_report()
    
    # Plot and save closeness centrality graph of top 10 characters
    create_closeness_centrality_plot(GRAPH, 10)
    print("# Plot and save closeness centrality graph of top 10 characters")
    get_status_report()
    
    # Plot and save character community detection graph
    create_character_community_detection(GRAPH)
    print("# Plot and save character community detection graph")
    get_status_report()
    
    # Clean up resources 
    clean_temp("./data/temp")
    print("# Clean up resources")
    get_status_report()


if __name__ == "__main__":
    # Load all the witcher characters 
    character_df = get_witcher_character_list()
    print("# Load all the witcher characters ")
    get_status_report()
    
    # Initialize empty list for graphs from books
    books_graph = []
    print("# Initialize empty list for graphs from books")
    get_status_report()
    
    # Get all witcher books stored in data/book file location
    all_books = get_all_witcher_books()
    print("# Get all witcher books stored in data/book file location")
    get_status_report()
    
    print("# Loop through book list and create graphs")
    get_status_report()
    # Loop through book list and create graphs
    for book in all_books:
        
        # Create sentence to character mapping and clean data
        sentence_entity_df_filtered = do_sentence_character_mapping(book, character_df)
        print(f"{book}: # Create sentence to character mapping and clean data")
        get_status_report()
        
        # Create relationship dataframe
        relationship_df = create_relationships(df = sentence_entity_df_filtered, window_size = 5)
        print(f"{book}: # Create relationship dataframe")
        get_status_report()
        
        # Calculate weight of each relation
        weighted_relationship_df = weighted_relationship_initial(relationship_df)
        print(f"{book}: # Calculate weight of each relation")
        get_status_report()
        
        # Store data frame to temp location
        weighted_relationship_df.to_csv(f'data/temp/{book.name}.csv', index=False)
        # Add book dataframe to centralized dataframe
        # all_books_relationship.append(relationship_df)
        print(f"{book}: # Store data frame to temp location")
        get_status_report()
        
        # Create a graph from a pandas dataframe
        GRAPH = nx.from_pandas_edgelist(weighted_relationship_df, 
                                    source = "source",
                                    target = "target",
                                    edge_attr  = "value",
                                    create_using = nx.Graph())
        print(f"{book}: # Create a graph from a pandas dataframe")
        get_status_report()
        
        # Collect each dataframe
        books_graph.append(GRAPH)
        print(f"{book}: # Collect each dataframe")
        get_status_report()
        
    # Start plotting all required graphs
    start_plotting(books_graph)
    print("# Start plotting all required graphs")
    get_status_report()
    

