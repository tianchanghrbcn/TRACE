#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Legacy implementation note."""
from __future__ import annotations 
import argparse ,json ,os ,time 
from pathlib import Path 

import numpy as np 
import optuna 
import pandas as pd 
from sklearn .cluster import DBSCAN 
from sklearn .metrics import silhouette_score ,davies_bouldin_score ,pairwise_distances 
from sklearn .preprocessing import StandardScaler 

# --------------------------- CLI --------------------------- #
cli =argparse .ArgumentParser ()
cli .add_argument ("--alpha",type =float ,default =0.5 )
cli .add_argument ("--beta",type =float ,default =None )# Legacy implementation note.
cli .add_argument ("--gamma",type =float ,default =None )# Legacy implementation note.
cli .add_argument ("--trials",type =int ,default =150 )
args ,_ =cli .parse_known_args ()
α =float (args .alpha )
if not (0.0 <=α <=1.0 ):
    raise ValueError ("--alpha text [0,1]")

    # Legacy implementation note.
csv_path =os .getenv ("CSV_FILE_PATH")
ds_id =os .getenv ("DATASET_ID")
algo_name =os .getenv ("ALGO")or "DBSCAN"
clean_tag =os .getenv ("CLEAN_STATE","raw")# raw / cleaned
if not csv_path :
    raise SystemExit ("CSV_FILE_PATH env text")
csv_path =os .path .normpath (csv_path )

# Legacy implementation note.
df =pd .read_csv (csv_path )
X =df [df .columns .difference ([c for c in df .columns if 'id'in c .lower ()])].copy ()
for c in X .columns :
    if X [c ].dtype .name in ("object","category"):
        X [c ]=X [c ].map (X [c ].value_counts (normalize =True ))
X =X .dropna ()
X_std =StandardScaler ().fit_transform (X )

start_t =time .time ()

# Legacy implementation note.
def _ci_harm (sil :float ,db :float )->float :
    """Legacy implementation note."""
    S =(sil +1.0 )/2.0 # [-1,1] → [0,1]
    D =1.0 /(1.0 +db )# [0,∞) → (0,1]
    eps =1e-12 
    return 1.0 /(α /max (S ,eps )+(1.0 -α )/max (D ,eps ))

def _evaluate (labels :np .ndarray ):
    n_clusters =len (np .unique (labels ))-(1 if -1 in labels else 0 )
    noise_r =float ((labels ==-1 ).mean ())
    if n_clusters <2 :
        return -np .inf ,np .nan ,np .nan ,noise_r 
    sil =silhouette_score (X_std ,labels )
    db =max (davies_bouldin_score (X_std ,labels ),1e-6 )
    comb =_ci_harm (sil ,db )
    return comb ,sil ,db ,noise_r 

    # Legacy implementation note.
def _objective (tr ):
    eps =tr .suggest_float ("eps",0.1 ,2.0 ,step =0.05 )
    ms =tr .suggest_int ("min_samples",5 ,50 )
    lb =DBSCAN (eps =eps ,min_samples =ms ).fit_predict (X_std )
    score ,_ ,_ ,noise =_evaluate (lb )
    return score *(1.0 -noise )# Legacy implementation note.

study =optuna .create_study (direction ="maximize")
study .optimize (_objective ,n_trials =args .trials ,show_progress_bar =False )
best =study .best_params 

# Legacy implementation note.
labels =DBSCAN (**best ).fit_predict (X_std )
comb ,sil ,db ,noise_r =_evaluate (labels )

# Legacy implementation note.
dmat =pairwise_distances (X_std )
core_mask =np .sum (dmat <=best ["eps"],axis =1 )>=best ["min_samples"]
stats ={
"combined":comb ,
"silhouette":sil ,
"davies_bouldin":db ,
"noise_ratio":noise_r ,
"core_count":int (core_mask .sum ()),
"border_count":int ((~core_mask &(labels !=-1 )).sum ()),
"noise_count":int ((labels ==-1 ).sum ()),
"neighbor_hist":np .bincount (
np .clip (np .sum (dmat <=best ["eps"],axis =1 ),0 ,49 )
).tolist (),
"best_eps":best ["eps"],
"best_min_samples":best ["min_samples"],
"weights":{"alpha":α },
"runtime_sec":time .time ()-start_t 
}

# Legacy implementation note.
base =Path (csv_path ).stem 
out_dir =Path .cwd ()/"results"/"clustered_data"/"DBSCAN"/f"clustered_{ds_id}"
out_dir .mkdir (parents =True ,exist_ok =True )

# Legacy implementation note.
(out_dir /f"{base}.txt").write_text (
"\n".join ([
f"Best parameters: eps={best['eps']}, min_samples={best['min_samples']}",
f"Final Combined Score: {comb}",
f"Final Silhouette Score: {sil}",
f"Final Davies-Bouldin Score: {db}"
]),encoding ="utf-8")

# Legacy implementation note.
sum_fp =out_dir /f"{base}_summary.json"
if sum_fp .exists ():
    whole =json .loads (sum_fp .read_text ())
else :
    whole ={}
whole ["combined"]=comb # Legacy implementation note.
whole .setdefault (clean_tag ,{}).update (stats )
sum_fp .write_text (json .dumps (whole ,indent =4 ),encoding ="utf-8")

print (f"All files saved in: {out_dir}")
print (f"Program completed in {stats['runtime_sec']:.2f} sec")
