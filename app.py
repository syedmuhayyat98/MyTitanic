from flask import Flask, render_template, request
import pickle
import numpy as np
import data_preparation as dp

model = pickle.load(open("titan_model.pkl","rb"))


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/subs", methods = ["POST"])
def result():
    if request.method == "POST":
        pclass = request.form["pclass"]
        age = request.form["age"]
        sibsp = request.form["sibsp"]
        parch = request.form["parch"]
        sex = request.form["sex"]
        embarked = request.form["embarked"]
        fare = request.form["fare"]
        fsize = int(sibsp) + int(parch) + 1
        sexm = dp.check_male(sex)
        sexf = dp.check_female(sex)
        emc = dp.check_emc(embarked)
        emq = dp.check_emq(embarked)
        ems = dp.check_ems(embarked)
        flow = dp.check_flow(fare)
        fmid = dp.check_fmedium(fare)
        fave = dp.check_fave(fare)
        fhigh = dp.check_fhigh(fare)
        features = np.array([pclass, age, sibsp, parch, fsize, sexf, sexm, emc, emq, ems, flow, fmid, fave, fhigh]).reshape(1, -1)
        prediction = model.predict(features)
        answer = dp.survival(prediction)

    return render_template("subs.html", p = answer)

if __name__ == "__main__":
    app.run(debug=True)