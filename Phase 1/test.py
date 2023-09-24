from preproccessing.preprocess import *
import warnings
warnings.filterwarnings("ignore")
import sys

def predict(df_file):
    #In real test file there won't be a y or ticketcategory

    df = pd.read_csv(df_file)

    y = df['price']
    y = y.str.replace(',', '')
    y = y.str.replace('.', '')
    y = y.astype(float)
    df.drop(['price'], axis=1, inplace=True)

    df = adjust_date(df)
    df = adjust_route(df)
    df = adjust_stop(df)
    df = adjust_time(df)

   
    cat_cols = ['airline', 'ch_code', 'stop', 'type', 'source', 'destination']
    with open('labelencoder.pickle', 'rb') as handle:
        le_dict2 = pickle.load(handle)
    for col in cat_cols:
        df[col] = le_dict2[col].transform(df[col])

    SI = pickle.load(open('simpleimputer.pkl','rb'))
    OHE = pickle.load(open('onehotencoder.pkl','rb'))
    SC = pickle.load(open('scaler.pkl','rb'))

    lr = pickle.load(open('linearregression.pkl','rb'))
    polymodel = pickle.load(open('polymodel.pkl','rb'))
    polyfeatures = pickle.load(open('polyfeatures.pkl','rb'))

    cat_cols = ['airline', 'ch_code', 'stop', 'type', 'month', 'source', 'destination']
    df[df.columns] = SI.transform(df[df.columns])
    df2 = df[cat_cols]
    df.drop(columns=cat_cols, inplace=True)
    df2 = OHE.transform(df2).toarray()
    df = df.to_numpy()
    new_df = np.concatenate((df, df2), axis=1)
    new_df = SC.transform(new_df)

    for name, model in zip(['lr', 'poly'],[lr, polymodel]):
        if name == 'poly':
            new_df = polyfeatures.transform(new_df)  
        y_pred = model.predict(new_df)
        print(f'Accuracy: {metrics.mean_squared_error(y, y_pred)}, name = {name}')
        print(f'r2: {metrics.r2_score(y, y_pred)}, name = {name}')

if __name__ == '__main__':
    predict(str(sys.argv[1]))