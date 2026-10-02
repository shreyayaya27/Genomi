"""Recover validation metrics from Round 2's saved first VQC model; does not retrain."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
from qiskit_machine_learning.algorithms.classifiers import VQC
from .run_controlled_validation_experiment import load_train_validation, metrics

RUN=Path("results/controlled_experiment/validation_round_02/zero_to_pi_cobyla_ansatz1")
def main():
    result=RUN/"result.json"
    if result.exists(): return
    _,_,_,X_val,y_val,_,_=load_train_validation()
    prep=joblib.load(RUN/"preprocessor.joblib");model=VQC.from_dill(str(RUN/"model.dill"))
    probability=model.predict_proba(prep.transform(X_val))[:,1];prediction=model.predict(prep.transform(X_val)).astype(int)
    circuit=model.feature_map.compose(model.ansatz)
    record={"configuration":"zero_to_pi_cobyla_ansatz1","status":"success_recovered","recovered_without_retraining":True,
            "training_seconds":None,"training_seconds_note":"Unavailable: original process crashed during post-training JSON serialization.",
            "optimizer_evaluations":None,"final_objective":None,"angle_range":"zero_to_pi","feature_map":"ZZFeatureMap","feature_map_reps":1,
            "ansatz":"RealAmplitudes","ansatz_reps":1,"entanglement":"linear","optimizer":"COBYLA","maxiter":200,"shots":1024,"simulator":"shot_aer","seed":42,"qubits":9,"circuit_depth":circuit.depth(),"ansatz_parameters":model.ansatz.num_parameters,**metrics(y_val,prediction,probability)}
    result.write_text(json.dumps(record,indent=2,default=lambda x:x.item() if hasattr(x,"item") else str(x))+"\n",encoding="utf-8")
if __name__=="__main__":main()
