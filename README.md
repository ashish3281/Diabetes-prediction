<h1>Diabetes Prediction using Machine Learning</h1>
<p>Diabetes, is a group of metabolic disorders in which there are high blood sugar levels over a prolonged period. Symptoms of high blood sugar include frequent urination, increased thirst, and increased hunger. If left untreated, diabetes can cause many complications. Acute complications can include diabetic ketoacidosis, hyperosmolar hyperglycemic state, or death. Serious long-term complications include cardiovascular disease, stroke, chronic kidney disease, foot ulcers, and damage to the eyes.

This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.
</p>
<h3>Objective</h3>
</p>We will try to build a machine learning model to accurately predict whether or not the patients in the dataset have diabetes or not?</p>
<h3>Details about the dataset:</h3>
<p>The datasets consists of several medical predictor variables and one target variable, Outcome. Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.</p>
<ul>
  <li>
Pregnancies: Number of times pregnant
  </li>
  <li>
    Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test</li>
  <li>
    BloodPressure: Diastolic blood pressure (mm Hg)</li>
  <li>
    SkinThickness: Triceps skin fold thickness (mm)</li>
  <li>
    Insulin: 2-Hour serum insulin (mu U/ml)</li>
  <li>
    BMI: Body mass index (weight in kg/(height in m)^2)</li>
  <li>
    DiabetesPedigreeFunction: Diabetes pedigree function</li>
  <li>
Age: Age (years) </li>
  <li>
Outcome: Class variable (0 or 1)
  </li>
  </ul>
<b>Number of Observation Units: 768</b>

<b>Variable Number: 9</b>

<b>Result; The model created as a result of XGBoost hyperparameter optimization became the model with the lowest Cross Validation Score value. (0.90)</b>
