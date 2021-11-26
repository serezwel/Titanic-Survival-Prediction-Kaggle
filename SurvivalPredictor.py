import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns
import matplotlib.pyplot as plt
#Function for age imputation based on class
def impute_age(cols):
    age = cols[0]
    Pclass = cols[1]
    if pd.isna(age):
        if Pclass == 1:
            return 38
        elif Pclass == 2:
            return 30
        else:
            return 25
    else:
        return age


def main():
    train_data = pd.read_csv("train.csv")
    test_data = pd.read_csv("test.csv")
    #EDA for checking empty values and imputation
    sns.heatmap(train_data.isna(), cmap='viridis', yticklabels=False, cbar=False)
    plt.savefig("CheckEmpty.png")
    plt.clf()
    sns.boxplot(x = "Pclass", y = "Age", data=train_data)
    plt.savefig("AgeCheck.png")
    plt.clf()
    #Imputation process
    imputing = train_data[["Age", "Pclass"]]    
    mean_age = imputing.groupby(["Pclass"]).mean().round()
    print(mean_age)
    train_data["Age"] = imputing.apply(impute_age, axis = 1)
    test_data["Age"] = test_data[["Age", "Pclass"]].apply(impute_age, axis = 1)

    X = train_data.copy()
    y = X.pop("Survived")
    
    test_result = test_data.loc[:, ["PassengerId"]]
    #Feature selection
    features_num = ["Pclass", "SibSp", "Parch", "Age", "Fare"]
    features_cat = ["Sex", "Embarked"]

    transformer_num = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
    )
    transformer_cat = make_pipeline(
        OneHotEncoder(handle_unknown="ignore")
    )
    preprocessor = make_column_transformer(
        (transformer_num, features_num),
        (transformer_cat, features_cat),
    )
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8)
    X_train = preprocessor.fit_transform(X_train)
    X_valid = preprocessor.transform(X_valid)
    test_data = preprocessor.transform(test_data)
    #There are 6 columns in X_train
    input_shape = [11]
    model = keras.Sequential([
        layers.BatchNormalization(input_shape = input_shape),
        layers.Dense(units=256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(rate=0.4),
        layers.Dense(units=256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(rate=0.4),
        layers.Dense(units=1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )
    early_stopping = keras.callbacks.EarlyStopping(
        patience=15,
        min_delta=0.001,
        restore_best_weights=True
    )
    history = model.fit(
        X_train, y_train,
        validation_data = (X_valid, y_valid),
        batch_size = 512,
        epochs=200,
        callbacks=[early_stopping],
    )
    Survived = model.predict(test_data)
    Survived = np.rint(Survived)
    Survived = Survived.astype(int)
    Survived_df = pd.DataFrame(Survived, columns=["Survived"])
    test_result = test_result.merge(Survived_df, left_index=True, right_index=True)
    test_result.to_csv("my_submission.csv", index=False)


if __name__ == "__main__":
    main()