import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import math
from datetime import datetime


# RQ1 functions


def describe_dataframe_from_large_jsonl(file_path, chunksize=200000):
    total_shape = (0, 0)
    dtypes = {}
    for chunk in pd.read_json(file_path, lines=True, chunksize=chunksize):
        total_shape = (total_shape[0] + chunk.shape[0], chunk.shape[1])
        dtypes.update(chunk.dtypes.to_dict())
    print("Total DataFrame Shape:", total_shape)
    print("\nColumn Data Types:")
    for column, dtype in dtypes.items():
        print(f"{column}: {dtype}")



# RQ2 functions

def plot_number_of_books(dataframe):
    """
    Plot the number of books for each author in descending order.
    
    Args:
        dataframe (DataFrame): A DataFrame containing author information.
    """
    # Select the first 50 rows of the DataFrame and the 'id' and 'book_ids' columns, then create a copy
    # This is done for a cleaner, more readable graph. To include all data, simply remove the iloc.
    selected_columns = dataframe.iloc[:50][['id', 'book_ids']].copy()
    selected_columns['num_books'] = selected_columns['book_ids'].apply(len)
    selected_columns = selected_columns.sort_values(by='num_books', ascending=False)
    number_of_books=selected_columns[['id', 'num_books']]
    number_of_books.plot(x='id', y='num_books', kind='bar', figsize=(20, 10))
    plt.xlabel('Author\'s id')
    plt.ylabel('Number of books')
    plt.title('Number of books for each author')
    plt.show()




def highest_number_of_rewiews(filename):
    """
    Find the book with the highest number of text reviews in a JSONLines file.
    
    Args:
        filename (str): The path to the JSONLines file to be processed.
    
    This function reads the JSONLines file in chunks and determines the book with the highest number of text reviews.
    It initializes variables to store the maximum value and the corresponding title. For each chunk, it calculates the maximum
    value in the 'text_reviews_count' column and finds the book title with that maximum value. If a higher maximum value is
    found, it updates the variables. Finally, it prints the maximum number of text reviews and the title of the book with
    that maximum number of reviews.
    """
    max_value = None
    corresponding_title = None

    for chunk in pd.read_json(filename, lines=True, chunksize=200000):
        max_in_chunk = chunk['text_reviews_count'].max()
        book_with_max_reviews = chunk[chunk['text_reviews_count'] == max_in_chunk]
        if max_value is None or max_in_chunk > max_value:
            max_value = max_in_chunk
            corresponding_title = book_with_max_reviews['title'].values[0]

    print(f"The max number of text reviews is: {max_value}")
    print(f"The book with {max_value} rewiews is: '{corresponding_title}'")



def highest_number_of_rewiews_bis(filename):
    """
    Find the book with the highest number of text reviews in a JSONLines file.
    
    Args:
        filename (str): The path to the JSONLines file to be processed.
    
    This function is another version of the last function, it reads the JSONLines file in chunks and identifies the book with the highest total number of text reviews.
    It ensures that case-insensitive book titles are considered as the titles are converted to lowercase. For each chunk, it groups the data by book title and calculates the total number of text reviews
    The function then determines the book with the highest number of reviews across all chunks and prints its details.

    Returns:
    - Prints the maximum number of text reviews and the corresponding book title.

    """
    max_value=0
    corresponding_title = None
    for chunk in pd.read_json(filename, lines=True, chunksize=200000):
        chunk['title'] = chunk['title'].str.lower()
        max_in_chunk = chunk.groupby('title')['text_reviews_count'].sum().reset_index()
        max_in_chunk = max_in_chunk.sort_values(by='text_reviews_count', ascending=False)
        if max_in_chunk.iloc[0]['text_reviews_count'] > max_value:
                max_value = max_in_chunk.iloc[0]['text_reviews_count']
                corresponding_title = max_in_chunk.iloc[0]['title']

    print(f"The max number of text reviews is: {max_value}")
    print(f"The book with {max_value} reviews is: '{corresponding_title}'")






def top_ten_worst_ten(filename):
    """
    Find and print the top ten and worst ten titles based on average ratings from a JSONLines file.
    
    Args:
        filename (str): The path to the JSONLines file to be processed.
    
    This function reads the JSONLines file in chunks, extracts 'title' and 'average_rating' columns, and sorts them by
    'average_rating' in descending order. For each chunk, it identifies the top ten and worst ten titles based on average
    ratings. The results from all chunks are concatenated to obtain the overall top ten and worst ten titles, which are
    then printed without row indices.
    """
    top_ten_list = []
    worst_ten_list = []  

    for chunk in pd.read_json(filename, lines=True, chunksize=10000):
        selected_columns = chunk[['title', 'average_rating']]
        selected_columns_sorted = selected_columns.sort_values(by='average_rating', ascending=False)

        top_ten_chunk = selected_columns_sorted.head(10)['title']
        top_ten_list.append(top_ten_chunk)

        worst_ten_chunk = selected_columns_sorted.tail(10)['title']
        worst_ten_list.append(worst_ten_chunk)

    top_ten = pd.concat(top_ten_list)
    worst_ten = pd.concat(worst_ten_list)

    print("Top ten titles:")
    for title in top_ten:
        print(title)

    print("Worst ten titles:")
    for title in worst_ten:
        print(title)


def distribution_of_languages(filename):
    for chunk in pd.read_json(filename, lines=True, chunksize=10000):
        chunk=chunk[chunk['language']!=""]
        chunk['language'] = chunk['language'].str.lower()
        selected_column=chunk[['language']].copy()
        chunk_language_count=selected_column['language'].value_counts()
        plt.figure(figsize=(16, 10))
        chunk_language_count.plot(kind='bar')
        plt.xlabel('Language')
        plt.ylabel('Number of Books')
        plt.title('Distribution of Books by Language')
        plt.show()
        break
        


def n_books_with_more_than_250_pages(filename):
    count = 0
    for chunk in pd.read_json(filename, lines=True, chunksize=10000):
        chunk = chunk[chunk['num_pages'] != '']
        count += chunk[chunk['num_pages'].astype(int) > 250].shape[0]
    print(count)


def distribution_of_fans_count(dataframe):
    selected_columns = dataframe[['id', 'book_ids', 'fans_count']].copy()
    selected_columns['num_books'] = selected_columns['book_ids'].apply(len)
    selected_columns = selected_columns.sort_values(by='num_books', ascending=False)
    most_prolific_authors=selected_columns.head(50)
    result=most_prolific_authors[['id', 'fans_count']]
    result.plot(x='id', y='fans_count', kind='bar', figsize=(20, 10))
    plt.xlabel('id')
    plt.ylabel('fans count')
    plt.title('Distribution of fans count')
    plt.show()




# RQ3 functions


def first_and_last_year_registered(filename):
    min_year=2023
    max_year=datetime.now().year
    for chunk in pd.read_json(filename, lines=True, chunksize=10000):
        chunk['original_publication_date'] = pd.to_datetime(chunk['original_publication_date'], errors='coerce')
        chunk = chunk.dropna(subset=['original_publication_date'])
        year_chunk = chunk['original_publication_date'].dt.year
        chunk_min_year=year_chunk.min()
        chunk_max_year=year_chunk.max()

        if chunk_min_year < min_year:
            min_year = chunk_min_year

        if chunk_max_year > max_year:
            if chunk_max_year >= datetime.now().year:
                max_year = datetime.now().year
            else:
                max_year = chunk_max_year

    return min_year, max_year




#RQ4 functions

def no_eponymous(dataframe):
    """
    Check if there are authors with precisely the same name in the dataset.
    Args:
        dataframe
    This function groups the authors by their name, counts the number of unique IDs for each name and so it checks if there are authors with the same name.
    """
    ret=True
    selected_columns = dataframe[['name', 'id']].copy()
    group=selected_columns.groupby('name')['id'].nunique().reset_index()
    for count in group['id']:
        if count>1:
            ret=False
            break
    if ret==True:
        print("It's true. There are no authors who have precisely the same name")
    else:
         print("It's false. There are authors who have precisely the same name")
        
   


def top_20_authors(dataframe, column1, column2):
    selected_columns = dataframe[[column1, column2]].copy()
    selected_columns_sorted=selected_columns.sort_values(by=column2, ascending=False)
    top_twenty = selected_columns_sorted.head(20)
    result_dataframe = top_twenty.rename(columns={column1: 'id'}).reset_index(drop=True)
    return result_dataframe

def longest_book_title(filename, dataframe):
    merged_df = pd.DataFrame(columns=['author_id', 'title'])
    for chunk in pd.read_json(filename, lines=True, chunksize=10000):
        selected_columns = chunk[['author_id', 'title']].copy()
        merged_chunk = selected_columns.merge(dataframe, left_on='author_id', right_on='id', how='inner')[['author_id', 'title']]
        if not merged_chunk.empty:
            merged_df = pd.concat([merged_df, merged_chunk], ignore_index=True)
    merged_df_sorted=merged_df.sort_values(by='title', key=lambda x: x.str.len(), ascending=False)
    longer_title=merged_df_sorted.iloc[0, 1]
    len_longer_title=len(longer_title)
    print(f"Longest book title among the books of the top 20 authors regarding their average rating: {longer_title}")
    print(f"Len of the longest title: {len_longer_title}")
    print("Is it the longest book title overall?")
    ret=True
    max_title=None
    for chunk in pd.read_json('/content/drive/MyDrive/lighter_books.json', lines=True, chunksize=10000):
      if chunk['title'].str.len().max() > len_longer_title:
        ret=False
        max_title = chunk.loc[chunk['title'].str.len().idxmax(), 'title']
        break
    if ret==False:
        print(f"No it's not. For example the book: {max_title} is longer")
    else:
        print("Yes it is")


def author_and_books(filename, id_list):
    author_books_dict = {}
    for author in id_list:
        author_books_dict[author]=[]
    for chunk in pd.read_json(filename, lines=True, chunksize=10000):
        for _, row in chunk.iterrows():
            author_id = row['author_id']
            if author_id in author_books_dict:
                book_title = row['title'].lower()
                if book_title not in author_books_dict[author_id]:
                    author_books_dict[author_id].append(book_title)
    for key, value in author_books_dict.items():
        print(f'Author id: {key}, Books: {value}')



def shortest_book_title(filename):
    shortest_title=None
    len_shortest_title=float('inf')
    for chunk in pd.read_json('/content/drive/MyDrive/lighter_books.json', lines=True, chunksize=10000):
        min_len = chunk['title'].str.len().min()
        if min_len < len_shortest_title:
            len_shortest_title = min_len
            shortest_title = chunk.loc[chunk['title'].str.len() == min_len, 'title'].values[0]
    print(f"The shortest title is: {shortest_title}")


def shortest_book_title_bis(filename):
    shortest_title=None
    len_shortest_title=float('inf')
    for chunk in pd.read_json('/content/drive/MyDrive/lighter_books.json', lines=True, chunksize=10000):
        min_len = chunk['title'].str.len().min()
        if min_len!=0 and min_len < len_shortest_title:
            len_shortest_title = min_len
            shortest_title = chunk.loc[chunk['title'].str.len() == min_len, 'title'].values[0]
    print(f"The shortest title is: {shortest_title}")


# RQ5 functions
# 1)

# Plot of the 10 most influent authors

def plot_df(my_df):
    '''
    This function plot the data frame
    :param my_df: the data frame we want to plot
    :return: nothing
    '''

    # Plot the fans count
    plt.figure(figsize=(10, 6))

    # Plot the authors with the respective fans count
    plt.bar(my_df['name'], my_df['fans_count'], label='Fans Count')

    plt.xlabel('Author')
    plt.ylabel('Fans Count')
    plt.title('Top Authors')
    plt.legend()
    plt.xticks(rotation=45)

    plt.show()


def plot_best_books(my_def):
    '''
    This function plot the data frame
    :param my_df: the data frame we want to plot
    :return: nothing
    '''
    # Create a new column 'book_ids_length' to store the length of each list
    my_def['book_ids_length'] = my_def['book_ids'].apply(len)



    plt.figure(figsize=(10, 6))

    # Plot the authors with the respective fans count
    plt.bar(my_def['name'], my_def['book_ids_length'], label='# of books')

    plt.xlabel('Author')
    plt.ylabel('Books Published')
    plt.title('Top Authors')
    plt.legend()
    plt.xticks(rotation=45)

    plt.show()

def get_top_books(my_df):
    '''
    This function extracts the books written by the top 10 influential authors.
    It reads and works on lighter_books.json file by chunks because the file is quite large.
    It extracts the books by chunks and stores them in the top_books df by concatenating the chunks.
    :param my_df: the data frame top_authors with the top 10 influential authors
    :return: a data frame named top_books containing all the books written by these authors.
    '''
    chunk_size = 1000

    # Create an empty DataFrame to store the top books
    top_books = pd.DataFrame()

    # Iterate through the chunks of lighter_books
    for chunk in pd.read_json('lighter_books.json', lines=True,
                              chunksize=chunk_size):
        # Filter the chunk to get books by top authors
        chunk_top_books = chunk[chunk['author_id'].isin(my_df['id'])]

        # Append the filtered chunk to the top_books DataFrame
        # top_books = top_books.concat(chunk_top_books, ignore_index = True)
        top_books = pd.concat([top_books, chunk_top_books], ignore_index=True)

    return top_books

# 2)

def longest_series_name(my_df):
    '''
    This function finds the longest series name written by the top 10 most influential authors.
    It finds it calculating the length of each series name of the books and storing it in a new column
     called 'series_name_length' and then finding the max across this column.
    :param my_df: data frame named top_books containing the books written by these authors
    :return: nothing
    '''

    #Make a copy in order to not modify the original one
    my_df_copy = my_df.copy()

    #Calculate the length of each series name adding a new column
    my_df_copy['series_name_length'] = my_df_copy['series_name'].apply(len)

    # Find the maximum length
    max_length = my_df_copy['series_name_length'].max()

    # Find the index of the row with the longest series name
    index_of_longest_series = my_df_copy[my_df_copy['series_name_length'] == max_length].index[0]

    # Retrieve the corresponding book row
    book_with_longest_series = my_df_copy.loc[index_of_longest_series]

    print("Longest Series Name:", book_with_longest_series['series_name'])

# 3)

def format_counts(my_df):
    '''
    The function extracts the different formats the books have been published counting
    the different type of format stored in the 'format' column of top_books and then normalizing the value.
    In order to make it clearer, it merges all the formats that are less than the 1% of the total.
    :param my_df: the data frame top_books
    :return: the data frame normalized_format_counts containing the books format distribution.
    '''

    # Count the frequency of each format
    format_counts = my_df['format'].value_counts()


    # Calculate the total count of formats
    total_count = format_counts.sum()

    # Define a threshold (for example 1% in this case)
    threshold = 0.01

    # Identify formats that are below the threshold
    formats_below_threshold = format_counts[(format_counts / total_count) < threshold]

    # Merge those formats into a category called "Others"
    format_counts[formats_below_threshold.index] = format_counts[formats_below_threshold.index].sum()
    format_counts = format_counts.drop(index=formats_below_threshold.index)


    # Normalize the frequencies to values between 0 and 1
    normalized_format_counts = format_counts / total_count

    # Add a category for "Others" with values below the threshold
    normalized_format_counts['Others'] = formats_below_threshold.sum() / total_count

    return normalized_format_counts


def chart_plot(my_df):
    plt.figure(figsize=(10, 6))
    my_df.plot(kind='bar')
    plt.title('Normalized Distribution of Book Formats')
    plt.xlabel('Format')
    plt.ylabel('Proportion')
    plt.xticks(rotation=45)
    plt.show()

# 4)

def plot_info(my_df,yaxis,ylabel):
    plt.figure(figsize=(10, 6))

    # Plot the authors with the respective average rating
    plt.bar(my_df['name'], my_df[yaxis], label=ylabel)

    plt.xlabel('Author')
    plt.ylabel(ylabel)
    plt.title('Top Authors')
    plt.legend()
    plt.xticks(rotation=45)

    plt.show()

def bias_plot(my_df,ylabel,title):
    plt.figure(figsize=(10, 6))
    my_df.plot(kind='bar', color=['red', 'blue', 'grey'])
    plt.title(title)
    plt.xlabel('Gender')
    plt.ylabel(ylabel)

    plt.xticks(rotation=0)
    plt.show()


# RQ6

def up_to_published_books(my_df, top_books):
    for index, author in my_df.iterrows():
        author_id = author['id']  # Get the author's ID
        # Filter the 'top_books' DataFrame to extract the books for the current author
        author_books = top_books[top_books['author_id'] == author_id].copy()

        author_name = author['name']  # Get the author name

        # Preprocess 'publication_date' to fill in missing month and day components
        author_books['publication_date'] = author_books['publication_date'].apply(
            lambda x: f"{x}-01-01" if len(str(x)) == 4 else x)

        # Convert the 'publication_date' column to a datetime object
        author_books['publication_date'] = pd.to_datetime(author_books['publication_date'], errors='coerce',
                                                          format='%Y-%m-%d')

        # Extract the year from the publication date and create a new column
        author_books['publication_year'] = author_books['publication_date'].dt.year

        # Group the data by publication year and count the number of books for each year
        yearly_counts = author_books['publication_year'].value_counts().sort_index()

        # Define the year bins for grouping
        year_bins = range(1960, 2031, 10)  # Adjust the range as needed

        # Create a new column 'year_bin' to categorize publication years into bins
        author_books['year_bin'] = pd.cut(author_books['publication_year'], bins=year_bins)

        # Group the data by year bins and count the number of books in each bin
        bin_counts = author_books['year_bin'].value_counts().sort_index()

        # Create a bar chart to visualize the number of books in each year bin
        plt.figure(figsize=(10, 6))
        bin_counts.plot(kind='bar')
        plt.title(f'Number of Books Published by Author {author_name} in Year Bins')
        plt.xlabel('Year Bins')
        plt.ylabel('Number of Books')
        plt.xticks(rotation=45)
        plt.show()









      
          
        
        
         







                
        
        
        
    
    


            
        


   
    





        






        

        

        
    
    
    
    
    
    
    
    
    
    


        



    
        
        
        
    












    
    







    







