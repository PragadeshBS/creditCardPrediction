import joblib
import pandas as pd

scaler_filename = "scaler.pkl"
scaler = joblib.load(scaler_filename)

model_filename = "model.pkl"
model = joblib.load(model_filename)


def get_prediction(
    age,
    experience,
    income,
    family,
    ccAvg,
    education,
    mortgage,
    personalLoan,
    securitiesAccount,
    cdAccount,
    online,
):
    cols = [
        "Age",
        "Experience",
        "Income",
        "Family",
        "CCAvg",
        "Education",
        "Mortgage",
        "PersonalLoan",
        "SecuritiesAccount",
        "CDAccount",
        "Online",
    ]
    df = pd.DataFrame(
        [
            [
                age,
                experience,
                income,
                family,
                ccAvg,
                education,
                mortgage,
                personalLoan,
                securitiesAccount,
                cdAccount,
                online,
            ]
        ],
        columns=cols,
    )
    scaled = scaler.transform(df)
    df_scaled = pd.DataFrame(scaled, columns=cols)
    return model.predict(df_scaled)[0]
