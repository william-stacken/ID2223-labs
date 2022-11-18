import os
import modal
import math
import numpy
from numpy import random
    
BACKFILL=False
LOCAL=False

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

def get_random_passenger():
    """
    Returns a DataFrame containing one random titanic passenger
    """
    import pandas as pd

    survived = round(numpy.random.uniform(2)) - 1

    if survived:
        df = pd.DataFrame({ "pclass": [round(numpy.random.uniform(3))],
                           "sex": [round(numpy.random.uniform(2)) - 1],
                           "age": [round(numpy.random.normal(32.756888, 16.765796) * 2) / 2],
                           "deck": [round(max(min(numpy.random.normal(1.518617, 1.998583), 7), 0))],
                           "family_size": [round(max(numpy.random.normal(1.976064, 1.339642), 1))],
                           "fare_per_person": [round(max(numpy.random.normal(27.168979, 41.494386), 0.0), 2)]
                          })
    else:
        df = pd.DataFrame({ "pclass": [round(numpy.random.uniform(2)) + 1],
                           "sex": [round(numpy.random.uniform(2)) - 1],
                           "age": [round(numpy.random.normal(32.756888, 8.765796) * 2) / 2],
                           "deck": [round(max(min(numpy.random.normal(1.518617, 1.998583), 7), 0))],
                           "family_size": [round(max(numpy.random.normal(1.0, 1.339642), 1))],
                           "fare_per_person": [round(max(numpy.random.normal(37.168979, 41.494386), 0.0), 2)]
                          })

    df['survived'] = survived

    return df



def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    if BACKFILL == True:
        titanic_df = pd.read_csv("https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv")

        titanic_df['survived'] = titanic_df['Survived']
        titanic_df['pclass'] = titanic_df['Pclass']
        titanic_df['deck'] = titanic_df['Cabin'].str.extract(r'([A-Z])?(\d)')[0]
	# Cast deck to int64 to ensure it is an bigint in hopsworks
        titanic_df['deck'] = numpy.int64(titanic_df['deck'].astype('category').cat.codes + 1)
        titanic_df['sex'] = titanic_df['Sex'].map(lambda x: sex_to_int(x))
        titanic_df['age'] = titanic_df['Age'].fillna(round(titanic_df['Age'].mean()))

        # Creating new family size column and normalizing fare by the family size
        titanic_df['family_size'] = titanic_df['SibSp'] + titanic_df['Parch'] + 1
        titanic_df['fare_per_person'] = round(titanic_df['Fare'] / (titanic_df['family_size']), 2)

        #print('   Mean Age = ' + str(t['age'].mean()))
        #print('Std Dev Age = ' + str(t['age'].std()))
        #t['age'] = (t['age'] - t['age'].mean()) / t['age'].std()

        # Remove unused columns and capitalized ones
        for col in ['Pclass', 'Age', 'Sex', 'Cabin', 'Name', 'Ticket', 'Embarked', 'SibSp', 'Parch', 'PassengerId', 'Fare', 'Survived']:
            titanic_df = titanic_df.drop(col, axis='columns')
    else:
        titanic_df = get_random_passenger()

    titanic_fg = fs.get_or_create_feature_group(
        name="titanic_modal",
        version=1,
        primary_key=["pclass","sex","age","deck","family_size","fare_per_person"],
        description="titanic passenger survival dataset")
    titanic_fg.insert(titanic_df, write_options={"wait_for_job" : True})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
