"""Predeclared Round 2 VQC validation experiment; test data is never loaded."""
from __future__ import annotations
import csv, json, time, traceback
from pathlib import Path
import joblib
import numpy as np
from qiskit.circuit.library import real_amplitudes, zz_feature_map
from qiskit.quantum_info import Statevector
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_aer.primitives import SamplerV2
from qiskit_machine_learning.algorithms.classifiers import VQC
from .run_controlled_validation_experiment import load_train_validation, balanced_indices, metrics, quantum_preprocessing
from .pilot_common import SEED

OUT = Path("results/controlled_experiment/validation_round_02")
GRID = [
    {"name": "zero_to_pi_cobyla_ansatz1", "optimizer": "COBYLA", "ansatz_reps": 1},
    {"name": "zero_to_pi_cobyla_ansatz2", "optimizer": "COBYLA", "ansatz_reps": 2},
    {"name": "zero_to_pi_spsa_ansatz1", "optimizer": "SPSA", "ansatz_reps": 1},
    {"name": "zero_to_pi_spsa_ansatz2", "optimizer": "SPSA", "ansatz_reps": 2},
]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X_train,y_train,train_ids,X_val,y_val,val_ids,assignment = load_train_validation()
    balanced = balanced_indices(y_train); X_train=X_train[balanced]; y_train=y_train[balanced]
    subset_path = OUT / "balanced_training_subset.csv"
    if not subset_path.exists():
        with subset_path.open("w",newline="",encoding="utf-8") as h:
            w=csv.DictWriter(h,fieldnames=["sample_id","case_id","label"]);w.writeheader()
            for position, i in enumerate(balanced): w.writerow({"sample_id":train_ids[i],"case_id":assignment[train_ids[i]]["case_id"],"label":int(y_train[position])})
    feature_map = zz_feature_map(9,reps=1,entanglement="linear")
    preflight = feature_map.compose(real_amplitudes(9,reps=2,entanglement="linear"))
    state = Statevector.from_instruction(preflight.assign_parameters({p:0.0 for p in preflight.parameters}))
    (OUT/"statevector_development_check.json").write_text(json.dumps({"mode":"exact_statevector_preflight","trained":False,"qubits":9,"circuit_depth":preflight.depth(),"statevector_norm":float(np.linalg.norm(state.data))},indent=2)+"\n")
    records=[]
    for config in GRID:
        run=OUT/config["name"];run.mkdir(exist_ok=True); history=[]
        existing = run / "result.json"
        if existing.exists():
            records.append(json.loads(existing.read_text(encoding="utf-8"))); continue
        record={"status":"failed","angle_range":"zero_to_pi","feature_map":"ZZFeatureMap","feature_map_reps":1,"ansatz":"RealAmplitudes","ansatz_reps":config["ansatz_reps"],"entanglement":"linear","optimizer":config["optimizer"],"maxiter":200,"shots":1024,"simulator":"shot_aer","seed":SEED,"qubits":9}
        try:
            prep=quantum_preprocessing("zero_to_pi").fit(X_train); xt,xv=prep.transform(X_train),prep.transform(X_val)
            algorithm_globals.random_seed=SEED
            fmap=zz_feature_map(9,reps=1,entanglement="linear"); ansatz=real_amplitudes(9,reps=config["ansatz_reps"],entanglement="linear")
            def cb(*args): history.append({"evaluation":len(history)+1,"objective":float(args[2] if len(args)>=3 else args[-1])})
            opt=COBYLA(maxiter=200) if config["optimizer"]=="COBYLA" else SPSA(maxiter=200,callback=cb)
            vqc=VQC(feature_map=fmap,ansatz=ansatz,optimizer=opt,sampler=SamplerV2(default_shots=1024,seed=SEED),callback=None if config["optimizer"]=="SPSA" else cb)
            start=time.perf_counter();vqc.fit(xt,y_train);elapsed=time.perf_counter()-start
            p=vqc.predict_proba(xv)[:,1];pred=vqc.predict(xv).astype(int);circuit=fmap.compose(ansatz)
            nfev=getattr(getattr(vqc,"fit_result",None),"nfev",None)
            record.update({"status":"success","training_seconds":elapsed,"optimizer_evaluations":nfev if nfev is not None else len(history),"final_objective":history[-1]["objective"] if history else None,"circuit_depth":circuit.depth(),"ansatz_parameters":ansatz.num_parameters,**metrics(y_val,pred,p)})
            joblib.dump(prep,run/"preprocessor.joblib");vqc.to_dill(str(run/"model.dill"))
        except Exception as e:
            record.update({"error_type":type(e).__name__,"error":str(e)});(run/"failure_traceback.txt").write_text(traceback.format_exc())
        record["configuration"]=config["name"]
        serializer=lambda value: value.item() if hasattr(value,"item") else str(value)
        (run/"loss_history.json").write_text(json.dumps(history,indent=2,default=serializer)+"\n");(run/"result.json").write_text(json.dumps(record,indent=2,default=serializer)+"\n");records.append(record)
    fields=sorted({k for r in records for k in r})
    with (OUT/"comparison_validation.csv").open("w",newline="",encoding="utf-8") as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(records)
    (OUT/"scope.json").write_text(json.dumps({"test_evaluated":False,"train_balanced":158,"validation_samples":len(y_val),"grid":GRID},indent=2)+"\n")
if __name__=="__main__":main()
