import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from functools import reduce

age = ctrl.Antecedent(np.arange(0, 101), 'Age')
smoke = ctrl.Antecedent(np.arange(0, 10.1, 0.1), 'Smoking')
cough = ctrl.Antecedent(np.arange(0, 81), 'Presistent Cough')
blood_cough = ctrl.Antecedent(np.arange(0, 80.1, 0.1), 'Coughing Up Blood')
weight_loss = ctrl.Antecedent(np.arange(0, 10.1, 0.1), 'Weight Loss')
shortness_breath = ctrl.Antecedent(np.arange(0, 80.1, 0.1), 'Shortness Of Breath')
bone_pains = ctrl.Antecedent(np.arange(0, 31), 'Bone Pains')
hoarseness_voice = ctrl.Antecedent(np.arange(0, 8.1, 0.1), 'Hoarseness Of Voice')
chest_pain = ctrl.Antecedent(np.arange(0, 81), 'Chest Pain')

age['Low'] = fuzz.trapmf(age.universe, [0, 0, 55, 60])
age['High'] = fuzz.trapmf(age.universe, [55, 60, 100, 100])

smoke['No'] = fuzz.trapmf(smoke.universe, [0, 0, 5, 5.2])
smoke['Yes'] = fuzz.trapmf(smoke.universe, [5, 5.2, 10, 10])

cough['Low'] = fuzz.trapmf(cough.universe, [0, 0, 14, 21])
cough['Medium'] = fuzz.trimf(cough.universe, [14, 31, 47])
cough['High'] = fuzz.trimf(cough.universe, [40, 58, 60])
cough['Very High'] = fuzz.trapmf(cough.universe, [58, 60, 80, 80])

blood_cough['Low'] = fuzz.trapmf(blood_cough.universe, [0, 0, 14, 21])
blood_cough['Medium'] = fuzz.trimf(blood_cough.universe, [14, 30.5, 47])
blood_cough['High'] = fuzz.trimf(blood_cough.universe, [40, 58, 60])
blood_cough['Very High'] = fuzz.trapmf(blood_cough.universe, [58, 60, 80, 80])

weight_loss['Low'] = fuzz.trimf(weight_loss.universe, [0, 0, 3])
weight_loss['Medium'] = fuzz.trimf(weight_loss.universe, [2.5, 4.5, 6])
weight_loss['High'] = fuzz.trimf(weight_loss.universe, [5, 6.5, 8])
weight_loss['Very High'] = fuzz.trimf(weight_loss.universe, [7.5, 10, 10])

shortness_breath['Low'] = fuzz.trapmf(shortness_breath.universe, [0, 0, 11, 14])
shortness_breath['Medium'] = fuzz.trimf(shortness_breath.universe, [10, 30.5, 47])
shortness_breath['High'] = fuzz.trimf(shortness_breath.universe, [30.5, 58, 60])
shortness_breath['Very High'] = fuzz.trapmf(shortness_breath.universe, [58, 60, 80, 80])

bone_pains['Low'] = fuzz.trapmf(bone_pains.universe, [0, 0, 12, 14])
bone_pains['High'] = fuzz.trapmf(bone_pains.universe, [12, 14, 30, 30])

hoarseness_voice['Low'] = fuzz.trapmf(hoarseness_voice.universe, [0, 0, 1, 4])
hoarseness_voice['Medium'] = fuzz.trapmf(hoarseness_voice.universe, [3, 4, 5, 8])
hoarseness_voice['High'] = fuzz.trimf(hoarseness_voice.universe, [4.5, 8, 8])

chest_pain['Low'] = fuzz.trapmf(chest_pain.universe, [0, 0, 7, 10])
chest_pain['Medium'] = fuzz.trimf(chest_pain.universe, [7, 25, 43])
chest_pain['High'] = fuzz.trimf(chest_pain.universe, [40, 53, 55])
chest_pain['Very High'] = fuzz.trapmf(chest_pain.universe, [53, 55, 80, 80])

stage = ctrl.Consequent(np.arange(0, 14.1, 0.1), 'Stage')
treatment = ctrl.Consequent(np.arange(0, 9.1, 0.1), 'Treatment')

stage['No Cancer'] = fuzz.trimf(stage.universe, [0, 0, 2.5])
stage['Stage 1'] = fuzz.trimf(stage.universe, [2, 5, 8])
stage['Stage 2'] = fuzz.trimf(stage.universe, [7.5, 10, 12])
stage['Stage 3'] = fuzz.trimf(stage.universe, [11.5, 14, 14])

treatment['No Treatment'] = fuzz.trimf(treatment.universe, [0, 0, 2.5])
treatment['Surgery'] = fuzz.trimf(treatment.universe, [2, 3, 5])
treatment['Chemotherapy'] = fuzz.trimf(treatment.universe, [4.5, 6, 7.5])
treatment['Radiation'] = fuzz.trimf(treatment.universe, [6.5, 9, 9])

stage_rule1 = [
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
]

treatment_rule1 = [
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
]

stage_rule2 = [
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['No Cancer']),
]

treatment_rule2 = [
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['No Treatment']),
]

stage_rule3 = [
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['No Cancer']),

    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['No Cancer']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['No Cancer']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['No Cancer']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['No Cancer']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['No Cancer']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['No Cancer']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], stage['No Cancer']),
]

treatment_rule3 = [
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['No Treatment']),

    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['No Treatment']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['No Treatment']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['No Treatment']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['No Treatment']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['No Treatment']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['No Treatment']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['No Treatment']),
]

stage_rule4 = [
    # stage 1
    ctrl.Rule((age['High'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 1']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 1']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 1']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 1']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 1']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 1']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 1']),

    # no cancer
    # age High
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['No Cancer']),

    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['No Cancer']),    

    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['No Cancer']),

    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['No Cancer']),

    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], stage['No Cancer']),

    # no cancer
    # age low
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['No Cancer']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['No Cancer']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['No Cancer']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], stage['No Cancer']),


    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['No Cancer']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['No Cancer']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], stage['No Cancer']),


    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['No Cancer']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], stage['No Cancer']),


    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], stage['No Cancer']),

    # no cancer
    # with cough
    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['No Cancer']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['No Cancer']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['No Cancer']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['No Cancer']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['No Cancer']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], stage['No Cancer']),

    # no cancer
    # with smoke
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['No Cancer']),

    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['No Cancer']),

    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['No Cancer']),

    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['No Cancer']),

    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['No Cancer']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['No Cancer']),

    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], stage['No Cancer']),
    
]

treatment_rule4 = [
    # Surgery
    ctrl.Rule((age['High'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['Surgery']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['Surgery']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['Surgery']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['Surgery']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['Surgery']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['Surgery']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Surgery']),

    # No Treatment
    # age High
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['No Treatment']),

    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['No Treatment']),    

    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['No Treatment']),

    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['No Treatment']),

    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['No Treatment']),

    # No Treatment
    # age low
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['No Treatment']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['No Treatment']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['No Treatment']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['No Treatment']),


    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['No Treatment']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['No Treatment']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['No Treatment']),


    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['No Treatment']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['No Treatment']),


    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['No Treatment']),

    # No Treatment
    # with cough
    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['No Treatment']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['No Treatment']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['No Treatment']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['No Treatment']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['No Treatment']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['No Treatment']),

    # No Treatment
    # with smoke
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['No Treatment']),

    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['No Treatment']),

    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['No Treatment']),

    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['No Treatment']),

    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['No Treatment']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['No Treatment']),

    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['No Treatment']),
]

stage_rule5 = [
    # Stage 2
    ctrl.Rule((age['High'] &  smoke['Yes'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 2']),
    ctrl.Rule((age['High'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 2']),
    ctrl.Rule((age['High'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 2']),
    ctrl.Rule((age['High'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 2']),
    ctrl.Rule((age['High'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 2']),
    ctrl.Rule((age['High'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 2']),
    
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 2']),
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 2']),
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 2']),
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 2']),
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 2']),
    
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 2']),
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 2']),
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 2']),
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 2']),
    
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 2']),
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 2']),
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 2']),
    
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 2']),
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 2']),
    
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 2']),


    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 2']),
    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 2']),
    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 2']),
    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 2']),
    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 2']),

    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 2']),
    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 2']),
    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 2']),
    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 2']),

    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 2']),
    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 2']),
    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 2']),

    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 2']),
    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 2']),

    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 2']),


    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 2']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 2']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 2']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 2']),

    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 2']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 2']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 2']),

    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 2']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 2']),

    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 2']),


    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 2']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 2']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 2']),

    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 2']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 2']),

    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 2']),


    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 2']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 2']),

    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 2']),


    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 2']),

    # Stage 1
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 1']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 1']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 1']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 1']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 1']),

    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 1']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 1']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 1']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 1']),

    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 1']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 1']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 1']),

    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 1']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 1']),

    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 1']),


    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 1']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 1']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 1']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 1']),

    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 1']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 1']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 1']),

    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 1']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 1']),

    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 1']),


    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 1']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 1']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 1']),

    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 1']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 1']),

    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 1']),


    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 1']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 1']),

    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 1']),


    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 1']),


    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 1']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 1']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 1']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 1']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 1']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 1']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 1']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 1']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 1']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 1']),


    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 1']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 1']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 1']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 1']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 1']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 1']),


    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 1']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 1']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 1']),


    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 1']),


    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 1']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 1']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 1']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 1']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 1']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 1']),


    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 1']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 1']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 1']),


    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 1']),


    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], stage['Stage 1']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 1']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 1']),


    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 1']),


    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 1']),

]

treatment_rule5 = [
    # Chemotherapy
    ctrl.Rule((age['High'] &  smoke['Yes'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['Chemotherapy']),
    ctrl.Rule((age['High'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['Chemotherapy']),
    ctrl.Rule((age['High'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['Chemotherapy']),
    ctrl.Rule((age['High'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['Chemotherapy']),
    ctrl.Rule((age['High'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['Chemotherapy']),
    ctrl.Rule((age['High'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Chemotherapy']),
    
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['Chemotherapy']),
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['Chemotherapy']),
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['Chemotherapy']),
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['Chemotherapy']),
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Chemotherapy']),
    
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['Chemotherapy']),
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['Chemotherapy']),
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['Chemotherapy']),
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Chemotherapy']),
    
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['Chemotherapy']),
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['Chemotherapy']),
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Chemotherapy']),
    
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['Chemotherapy']),
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Chemotherapy']),
    
    ctrl.Rule((age['High'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Chemotherapy']),


    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['Chemotherapy']),
    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['Chemotherapy']),
    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['Chemotherapy']),
    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['Chemotherapy']),
    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Chemotherapy']),

    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['Chemotherapy']),
    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['Chemotherapy']),
    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['Chemotherapy']),
    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Chemotherapy']),

    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['Chemotherapy']),
    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['Chemotherapy']),
    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Chemotherapy']),

    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['Chemotherapy']),
    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Chemotherapy']),

    ctrl.Rule((age['High'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Chemotherapy']),


    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['Chemotherapy']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['Chemotherapy']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['Chemotherapy']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Chemotherapy']),

    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['Chemotherapy']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['Chemotherapy']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Chemotherapy']),

    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['Chemotherapy']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Chemotherapy']),

    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Chemotherapy']),


    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['Chemotherapy']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['Chemotherapy']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Chemotherapy']),

    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['Chemotherapy']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Chemotherapy']),

    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Chemotherapy']),


    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['Chemotherapy']),
    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Chemotherapy']),

    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Chemotherapy']),


    ctrl.Rule((age['High'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Chemotherapy']),

    # Surgery
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['Surgery']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['Surgery']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['Surgery']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['Surgery']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Surgery']),

    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['Surgery']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['Surgery']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['Surgery']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Surgery']),

    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['Surgery']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['Surgery']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Surgery']),

    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['Surgery']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Surgery']),

    ctrl.Rule((age['Low'] &  smoke['Yes'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Surgery']),


    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['Surgery']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['Surgery']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['Surgery']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Surgery']),

    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['Surgery']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['Surgery']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Surgery']),

    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['Surgery']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Surgery']),

    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Surgery']),


    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['Surgery']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['Surgery']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Surgery']),

    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['Surgery']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Surgery']),

    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Surgery']),


    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['Surgery']),
    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Surgery']),

    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Surgery']),


    ctrl.Rule((age['Low'] &  smoke['Yes'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Surgery']),


    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['Surgery']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['Surgery']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['Surgery']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Surgery']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['Surgery']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['Surgery']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Surgery']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['Surgery']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Surgery']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Surgery']),


    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['Surgery']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['Surgery']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Surgery']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['Surgery']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Surgery']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Surgery']),


    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['Surgery']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Surgery']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Surgery']),


    ctrl.Rule((age['Low'] &  smoke['No'] &  ~cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Surgery']),


    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & chest_pain['Low'], treatment['Surgery']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['Surgery']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Surgery']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['Surgery']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Surgery']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Surgery']),


    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['Surgery']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Surgery']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Surgery']),


    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & ~blood_cough['Low']) & weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Surgery']),


    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & chest_pain['Low'], treatment['Surgery']),
    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Surgery']),

    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Surgery']),


    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & ~weight_loss['Low'] & shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Surgery']),


    ctrl.Rule((age['Low'] &  smoke['No'] &  cough['Low'] & blood_cough['Low']) & weight_loss['Low'] & ~shortness_breath['Low'] & bone_pains['High'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Surgery']),

]

stage_rule6 = [
    ctrl.Rule((~age['Low'] &  ~smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & ~bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], stage['Stage 3']),
]

treatment_rule6 = [
    ctrl.Rule((~age['Low'] &  ~smoke['No'] &  ~cough['Low'] & ~blood_cough['Low']) & ~weight_loss['Low'] & ~shortness_breath['Low'] & ~bone_pains['Low'] & ~hoarseness_voice['Low'] & ~chest_pain['Low'], treatment['Radiation']),
]

stage_rules = stage_rule1 + stage_rule2 + stage_rule3 + stage_rule4 + stage_rule5 + stage_rule6
treatment_rules = treatment_rule1 + treatment_rule2 + treatment_rule3 + treatment_rule4 + treatment_rule5 + treatment_rule6

stage_ctrl = ctrl.ControlSystem(stage_rules)
stage_sim = ctrl.ControlSystemSimulation(stage_ctrl)

treatment_ctrl = ctrl.ControlSystem(treatment_rules)
treatment_sim = ctrl.ControlSystemSimulation(treatment_ctrl)

# errors = []
def predict(
    age,
    smoke,
    cough,
    blood_cough,
    weight_loss,
    shortness_breath,
    bone_pains,
    hoarseness_voice,
    chest_pain,
    stage_sim = stage_sim,
    treatment_sim = treatment_sim
):
    is_catch_error = 0
    try:
        stage_sim.input['Age'] = age
        stage_sim.input['Smoking'] = smoke
        stage_sim.input['Presistent Cough'] = cough
        stage_sim.input['Coughing Up Blood'] = blood_cough
        stage_sim.input['Weight Loss'] = weight_loss
        stage_sim.input['Shortness Of Breath'] = shortness_breath
        stage_sim.input['Bone Pains'] = bone_pains
        stage_sim.input['Hoarseness Of Voice'] = hoarseness_voice
        stage_sim.input['Chest Pain'] = chest_pain
        stage_sim.compute()
        stage_output = stage_sim.output['Stage']
    except Exception as e:
        # error = f"Error in compute stage: {e}"
        # if error not in errors:
        #     # print(error)
        #     # print(f"traceback: {traceback.format_exc()}")
        #     # print(f"{age = },{smoke = },{cough = },{blood_cough = },{weight_loss = },{shortness_breath = },{bone_pains = },{hoarseness_voice = },{chest_pain = }\n")
        #     errors.append(error)
        stage_output = 10
        is_catch_error += 1 

    try:   
        treatment_sim.input['Age'] = age
        treatment_sim.input['Smoking'] = smoke
        treatment_sim.input['Presistent Cough'] = cough
        treatment_sim.input['Coughing Up Blood'] = blood_cough
        treatment_sim.input['Weight Loss'] = weight_loss
        treatment_sim.input['Shortness Of Breath'] = shortness_breath
        treatment_sim.input['Bone Pains'] = bone_pains
        treatment_sim.input['Hoarseness Of Voice'] = hoarseness_voice
        treatment_sim.input['Chest Pain'] = chest_pain
        treatment_sim.compute()
        treatment_output = treatment_sim.output['Treatment']
    except Exception as e:
        treatment_output = 6
        is_catch_error += 2

    return stage_output, treatment_output, is_catch_error

def output_fuzzification(stage_output, treatment_output, stage=stage, treatment=treatment):
    
    no_cancer_membership = fuzz.interp_membership(stage.universe, stage['No Cancer'].mf, stage_output)
    stage1_membership = fuzz.interp_membership(stage.universe, stage['Stage 1'].mf, stage_output)
    stage2_membership = fuzz.interp_membership(stage.universe, stage['Stage 2'].mf, stage_output)
    stage3_membership = fuzz.interp_membership(stage.universe, stage['Stage 3'].mf, stage_output)

    no_treatment_membership = fuzz.interp_membership(treatment.universe, treatment['No Treatment'].mf, treatment_output)
    surgery_membership = fuzz.interp_membership(treatment.universe, treatment['Surgery'].mf, treatment_output)
    chemotherapy_membership = fuzz.interp_membership(treatment.universe, treatment['Chemotherapy'].mf, treatment_output)
    radiation_membership = fuzz.interp_membership(treatment.universe, treatment['Radiation'].mf, treatment_output)

    return no_cancer_membership, stage1_membership, stage2_membership, stage3_membership, no_treatment_membership, surgery_membership, chemotherapy_membership, radiation_membership


def get_membership_predict(
        age,
        smoke,
        cough,
        blood_cough,
        weight_loss,
        shortness_breath,
        bone_pains,
        hoarseness_voice,
        chest_pain
    ):
    stage_output, treatment_output, _ = predict(
        age = age,
        smoke = smoke,
        cough = cough,
        blood_cough = blood_cough,
        weight_loss = weight_loss,
        shortness_breath = shortness_breath,
        bone_pains = bone_pains,
        hoarseness_voice = hoarseness_voice,
        chest_pain = chest_pain
    )
    values = output_fuzzification(
        stage_output,
        treatment_output
        )
    keys = ["no_cancer_membership", "stage1_membership", "stage2_membership", "stage3_membership","no_treatment_membership", "surgery_membership", "chemotherapy_membership", "radiation_membership"]
    return dict(zip(keys, values))