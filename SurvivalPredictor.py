import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt


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

    sns.heatmap(train_data.isna(), cmap='viridis', yticklabels=False, cbar=False)
    plt.savefig("CheckEmpty.png")
    plt.clf()
    sns.boxplot(x = "Pclass", y = "Age", data=train_data)
    plt.savefig("AgeCheck.png")
    plt.clf()
    y = train_data["Survived"]

    imputing = train_data[["Age", "Pclass"]]
    mean = imputing.groupby(["Pclass"]).mean().round()
    print(mean)
    train_data["Age"] = imputing.apply(impute_age, axis = 1)
    test_data["Age"] = test_data[["Age", "Pclass"]].apply(impute_age, axis = 1)

    features = ["Pclass", "Sex", "SibSp", "Parch", "Age"]
    X = pd.get_dummies(train_data[features])
    X_test = pd.get_dummies(test_data[features])

    model = RandomForestClassifier(n_estimators = 100, max_depth = 5, random_state = 1)
    model.fit(X, y)
    predictions = model.predict(X_test)

    output = pd.DataFrame({"PassengerId": test_data.PassengerId, "Survived": predictions})
    output.to_csv("my_submission.csv", index=False)

if __name__ == "__main__":
    main()