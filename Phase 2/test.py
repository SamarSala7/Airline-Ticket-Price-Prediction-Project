from preproccessing.preprocess import *
import warnings
warnings.filterwarnings('ignore')
import sys

def predict(df_file):
    #In real test file there won't be a y or ticketcategory

    labels = {
    'cheap':0,
    'moderate':1,
    'expensive':2,
    'very expensive':3
    }

    df = pd.read_csv(df_file)

    df = adjust_date(df)
    df = adjust_route(df)
    df = adjust_stop(df)
    df = adjust_time(df)

    y = df['TicketCategory']
    y = y.map(labels)
    df.drop(['TicketCategory'], axis=1, inplace=True)

    cat_cols = ['airline', 'ch_code', 'stop', 'type', 'source', 'destination']
    with open('labelencoder.pickle', 'rb') as handle:
        le_dict2 = pickle.load(handle)
    for col in cat_cols:
        df[col] = le_dict2[col].transform(df[col])

    SI = pickle.load(open('simpleimputer.pkl','rb'))
    OHE = pickle.load(open('onehotencoder.pkl','rb'))
    SC = pickle.load(open('scaler.pkl','rb'))
    xgb = pickle.load(open('xgb.pkl','rb'))
    lgbm = pickle.load(open('lgbm.pkl','rb'))
    rf = pickle.load(open('rf.pkl','rb'))

    cat_cols = ['airline', 'ch_code', 'stop', 'type', 'month', 'source', 'destination']
    df[df.columns] = SI.transform(df[df.columns])
    df2 = df[cat_cols]
    df.drop(columns=cat_cols, inplace=True)
    df2 = OHE.transform(df2).toarray()
    df = df.to_numpy()
    new_df = np.concatenate((df, df2), axis=1)
    new_df = SC.transform(new_df)

    for name, model in zip(['xgb', 'lgbm', 'rf'],[xgb, lgbm, rf]):
        y_pred = model.predict(new_df)
        print(f'Accuracy: {metrics.accuracy_score(y, y_pred)*100}%, models name is {name}')

if __name__ == '__main__':
    predict(str(sys.argv[1]))