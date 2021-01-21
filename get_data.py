import pandas as pd

def get_data():
    data = pd.read_csv('/Users/royakash/Documents/GitHub/MRR-SodaLimeSilicaGlass/L27 Orthogonal array for MRR.csv')
    train_data = data.sample(frac=0.65, random_state=42)
    test_data = data.drop(train_data.index)

    train_features = train_data.copy()
    test_features = test_data.copy()

    train_features.pop('Expt Νο.')
    test_features.pop('Expt Νο.')

    train_labels = pd.concat([train_features.pop(x) for x in ['MRR', 'Ra']], axis=1)
    test_labels = pd.concat([test_features.pop(x) for x in ['MRR', 'Ra']], axis=1)

    return (train_features, train_labels), (test_features, test_labels)