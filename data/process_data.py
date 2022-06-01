import sys
# import libraries
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    INPUT:
    messages_filepath(str) - the path of the disaster_messages.csv file
    categories_filepath(str) - the path of the disaster_categories.csv file
    OUTPUT:
    df(dataframe) - dataframe of messages table and categories table merged on id
    
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,how="left",on="id")
    return df

def clean_data(df):
    '''
    INPUT:
    df dataframe - dataframe resulting from merging of messages and categories by id
   
    OUTPUT:
    df dataframe - dateframe with categories manipulated and cleaned
    
    '''
# create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(";",expand=True)
# select the first row of the categories dataframe
    row =categories.iloc[:1,:]
# use this row to extract a list of new column names for categories.
# one way is to apply a lambda function that takes everything 
# up to the second to last character of each string with slicing
    extract_list = lambda x:x[0][:-2]
    category_colnames = list(row.apply(extract_list))

# rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:    
    # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x:x[-1])
       # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    for i in categories.columns:
      
    
## fix any values that have not got 0 or 1 for the binary variables    
        categories.loc[(categories[i]!=0) & (categories[i]!=1) ,i] = 1
   
           
  
 # drop the original categories column from `df`
    df.drop(labels="categories",axis=1,inplace=True)

# concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)

    df=df.drop_duplicates()
    return df    



def save_data(df, database_filename):
    '''
    INPUT:
    df dataframe - cleaned dataset
   
    OUTPUT:
    df dataframe output as sql file
    
    '''
    engine = create_engine('sqlite:///STOTAJO.db')
    #df.drop('MESSAGE_CATEGORIES',if_exists='replace',engine)
    df.to_sql('MESSAGE_CATEGORIES', engine,if_exists='replace', index=False)
    pass  


def main():
    '''
    Puts all code above together and outputs the data
    INPUT:
    df dataframe - initial csv files
   
    OUTPUT:
    df dataframe in sql - dateframe with categories manipulated and cleaned
    
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
    
