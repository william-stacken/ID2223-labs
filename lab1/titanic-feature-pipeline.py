import os
import modal
import math
import numpy
    
BACKFILL=True
LOCAL=True

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4","joblib","seaborn","sklearn","dataframe-image"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()

def cabin_to_deck(cabin):
    for deck in ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G']:
        if deck in str(cabin):
            return ord(deck)
    return math.nan

def sex_to_int(val):
    asstr = str(val).lower()
    if asstr == 'male':
        return 0
    elif asstr == 'female':
        return 1
    raise ValueError('Uexpected value "' + asstr + '"')


def generate_flower(survived, sepal_len_max, sepal_len_min, sepal_width_max, sepal_width_min, 
                    petal_len_max, petal_len_min, petal_width_max, petal_width_min):
    """
    Returns a single titanic flower as a single row in a DataFrame
    """
    import pandas as pd
    import random

    df = pd.DataFrame({ "sepal_length": [random.uniform(sepal_len_max, sepal_len_min)],
                       "sepal_width": [random.uniform(sepal_width_max, sepal_width_min)],
                       "petal_length": [random.uniform(petal_len_max, petal_len_min)],
                       "petal_width": [random.uniform(petal_width_max, petal_width_min)]
                      })
    df['variety'] = name
    return df


def get_random_passenger():
    """
    Returns a DataFrame containing one random titanic flower
    """
    import pandas as pd
    import random

    virginica_df = generate_flower("Virginica", 8, 5.5, 3.8, 2.2, 7, 4.5, 2.5, 1.4)
    versicolor_df = generate_flower("Versicolor", 7.5, 4.5, 3.5, 2.1, 3.1, 5.5, 1.8, 1.0)
    setosa_df =  generate_flower("Setosa", 6, 4.5, 4.5, 2.3, 1.2, 2, 0.7, 0.3)

    # randomly pick one of these 3 and write it to the featurestore
    pick_random = random.uniform(0,3)
    if pick_random >= 2:
        titanic_df = virginica_df
        print("Virginica added")
    elif pick_random >= 1:
        titanic_df = versicolor_df
        print("Versicolor added")
    else:
        titanic_df = setosa_df
        print("Setosa added")

    return titanic_df



def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    if BACKFILL == True:
        titanic_df = pd.read_csv("https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv")
    else:
        titanic_df = get_random_passenger()

    #titanic_df['Deck'] = titanic_df['Cabin'].map(lambda x: cabin_to_deck(x))
    titanic_df['Deck'] = titanic_df['Cabin'].str.extract(r'([A-Z])?(\d)')[0]
    titanic_df['Deck'] = titanic_df['Deck'].astype('category').cat.codes + 1
    titanic_df['Sex'] = titanic_df['Sex'].map(lambda x: sex_to_int(x))
    titanic_df['Age'] = titanic_df['Age'].fillna(round(titanic_df['Age'].mean()))

    # Creating new family size column and normalizing fare by the family size
    titanic_df['family_size'] = titanic_df['SibSp'] + titanic_df['Parch'] + 1
    titanic_df['fare_per_person'] = titanic_df['Fare'] / (titanic_df['family_size'])

    #print('   Mean Age = ' + str(t['age'].mean()))
    #print('Std Dev Age = ' + str(t['age'].std()))
    #t['age'] = (t['age'] - t['age'].mean()) / t['age'].std()

    # Remove unused columns
    for col in ['Cabin', 'Name', 'Ticket', 'Embarked', 'SibSp', 'Parch', 'PassengerId', 'Fare']:
        titanic_df = titanic_df.drop(col, axis='columns')

    titanic_fg = fs.get_or_create_feature_group(
        name="titanic_modal",
        version=1,
        primary_key=["Pclass","Sex","Age","FamilySize","Deck","family_size","fare_per_person"], 
        description="titanic passenger survival dataset")
    titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
