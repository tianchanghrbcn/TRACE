#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Legacy implementation note."""
import os ,time ,math ,json 
import numpy as np 
import pandas as pd 
import optuna 
from pathlib import Path 
from sklearn .cluster import AgglomerativeClustering 
from sklearn .preprocessing import StandardScaler 
from sklearn .metrics import (silhouette_score ,
davies_bouldin_score ,
pairwise_distances )

# --------------------------------------------------
# Legacy implementation note.
# --------------------------------------------------
csv_file_path =os .getenv ("CSV_FILE_PATH")
dataset_id =os .getenv ("DATASET_ID")
algorithm_name =os .getenv ("ALGO")
clean_state =os .getenv ("CLEAN_STATE","raw")# raw / cleaned

if not csv_file_path :
    raise SystemExit ("CSV_FILE_PATH not provided")
csv_file_path =os .path .normpath (csv_file_path )

# --------------------------------------------------
# Legacy implementation note.
# --------------------------------------------------
df =pd .read_csv (csv_file_path )
excluded =[c for c in df .columns if 'id'in c .lower ()]
X =df [df .columns .difference (excluded )].copy ()

for col in X .columns :
    if X [col ].dtype in ("object","category"):
        X [col ]=X [col ].map (X [col ].value_counts (normalize =True ))
X =X .dropna ()
# Legacy implementation note.
used_index =X .index .copy ()

X_scaled =StandardScaler ().fit_transform (X )

# Legacy implementation note.
alpha =0.47 
beta =1 -alpha 
gamma =0.00 

# Legacy implementation note.
global_centroid =X_scaled .mean (axis =0 )
SSE_max =float (np .sum ((X_scaled -global_centroid )**2 ))

start_time =time .time ()

# --------------------------------------------------
# Legacy implementation note.
# --------------------------------------------------
def _sse (labels :np .ndarray )->float :
    """Legacy implementation note."""
    sse =0.0 
    for lbl in np .unique (labels ):
        pts =X_scaled [labels ==lbl ]
        centroid =pts .mean (axis =0 )
        sse +=np .sum ((pts -centroid )**2 )
    return float (sse )

def combined (db ,sil ,sse ):
    S =(sil +1.0 )/2.0 # [-1,1] → [0,1]
    D =1.0 /(1.0 +db )# [0,∞) → (0,1]
    eps =1e-12 
    return 1.0 /(alpha /max (S ,eps )+beta /max (D ,eps ))

def run_hc_tracking (k :int ,linkage :str ,metric :str ):
    """Legacy implementation note."""
    hc =AgglomerativeClustering (
    n_clusters =k ,
    linkage =linkage ,
    affinity =metric ,
    compute_distances =True # sklearn ≥1.2
    )
    labels =hc .fit_predict (X_scaled )

    merges =[]
    if hasattr (hc ,"children_")and hasattr (hc ,"distances_"):
        for step ,(i ,j ,d )in enumerate (
        zip (hc .children_ [:,0 ],hc .children_ [:,1 ],hc .distances_ )):
            merges .append ({"step":int (step +1 ),
            "cluster_i":int (i ),
            "cluster_j":int (j ),
            "dist":float (d )})

            # Legacy implementation note.
    dist_mat =pairwise_distances (X_scaled ,metric ="euclidean")
    intra ,inter ,cnt_intra ,cnt_inter =0.0 ,0.0 ,0 ,0 
    for i in range (len (labels )):
        for j in range (i +1 ,len (labels )):
            if labels [i ]==labels [j ]:
                intra +=dist_mat [i ,j ];cnt_intra +=1 
            else :
                inter +=dist_mat [i ,j ];cnt_inter +=1 
    intra_mean =intra /max (cnt_intra ,1 )
    inter_mean =inter /max (cnt_inter ,1 )
    core_stats ={"intra_dist_mean":intra_mean ,
    "inter_dist_mean":inter_mean ,
    "ratio_intra_inter":intra_mean /(inter_mean +1e-12 )}
    return labels ,merges ,core_stats 

    # --------------------------------------------------
    # Legacy implementation note.
    # --------------------------------------------------
optuna_trials =[]

def objective (trial ):
    t0 =time .time ()# Legacy implementation note.
    k =trial .suggest_int ("n_clusters",5 ,max (5 ,math .isqrt (X .shape [0 ])))
    linkage =trial .suggest_categorical ("linkage",
    ["ward","complete","average","single"])
    metric =trial .suggest_categorical ("metric",
    ["euclidean","manhattan","cosine"])
    if linkage =="ward"and metric !="euclidean":
        raise optuna .exceptions .TrialPruned ()

    labels ,merges ,core_stats =run_hc_tracking (k ,linkage ,metric )
    sil =silhouette_score (X_scaled ,labels )
    db =davies_bouldin_score (X_scaled ,labels )
    sse =_sse (labels )
    comb =combined (db ,sil ,sse )
    h_max =merges [-1 ]["dist"]if merges else 0.0 

    optuna_trials .append ({
    "trial_number":int (trial .number ),
    "n_clusters":int (k ),
    "linkage":str (linkage ),
    "metric":str (metric ),
    "combined_score":float (comb ),
    "silhouette":float (sil ),
    "davies_bouldin":float (db ),
    "sse":float (sse ),
    "n_merge_steps":int (len (merges )),
    "h_max":float (h_max ),# Legacy implementation note.
    **{k :(float (v )if isinstance (v ,(int ,float ,np .floating ))else float (v ))
    for k ,v in {
    "intra_dist_mean":core_stats ["intra_dist_mean"],
    "inter_dist_mean":core_stats ["inter_dist_mean"],
    "ratio_intra_inter":core_stats ["ratio_intra_inter"]
    }.items ()},
    "trial_duration_sec":float (time .time ()-t0 )# NEW
    })
    return comb 

optuna .create_study (direction ="maximize").optimize (objective ,n_trials =100 )
best =max (optuna_trials ,key =lambda d :d ["combined_score"])

# --------------------------------------------------
# Legacy implementation note.
# --------------------------------------------------
final_best_k =best ["n_clusters"]
linkage_optuna =best ["linkage"]
metric_optuna =best ["metric"]
best_cov_type =f"{linkage_optuna}-{metric_optuna}"# Legacy implementation note.

labels_final ,merge_history ,core_stats_final =run_hc_tracking (
final_best_k ,linkage_optuna ,metric_optuna )

final_db =davies_bouldin_score (X_scaled ,labels_final )
final_sil =silhouette_score (X_scaled ,labels_final )
final_sse =_sse (labels_final )
final_comb =combined (final_db ,final_sil ,final_sse )
final_hmax =merge_history [-1 ]["dist"]if merge_history else 0.0 # ★

# Legacy implementation note.
unique_labels =np .unique (labels_final )
cluster_sizes =[int (np .sum (labels_final ==c ))for c in unique_labels ]
cluster_centers =[]
for c in unique_labels :
    pts =X_scaled [labels_final ==c ]
    centroid =pts .mean (axis =0 )
    cluster_centers .append ([float (x )for x in centroid ])

    # --------------------------------------------------
    # Legacy implementation note.
    # --------------------------------------------------
base =Path (csv_file_path ).stem 
root =Path .cwd ()/".."/".."/".."/"results"/"clustered_data"/"HC"/algorithm_name /f"clustered_{dataset_id}"
root .mkdir (parents =True ,exist_ok =True )

# Legacy implementation note.
(Path (root )/f"{base}.txt").write_text (
"\n".join ([
f"Best parameters: n_components={final_best_k}, covariance type={best_cov_type}",
f"Final Combined Score: {final_comb}",
f"Final Silhouette Score: {final_sil}",
f"Final Davies-Bouldin Score: {final_db}"
]),
encoding ="utf-8"
)

# Legacy implementation note.
with open (root /f"{base}_{clean_state}_merge_history.json","w",encoding ="utf-8")as fp :
    json .dump (merge_history ,fp ,indent =4 )

summary ={
"clean_state":clean_state ,
"best_k":final_best_k ,
"linkage":linkage_optuna ,
"metric":metric_optuna ,
"combined":final_comb ,
"silhouette":final_sil ,
"davies_bouldin":final_db ,
"sse":final_sse ,
"weights":{"alpha":alpha ,"beta":beta ,"gamma":gamma },
**core_stats_final ,
"n_merge_steps":len (merge_history ),
"h_max":final_hmax ,# Legacy implementation note.
"total_runtime_sec":time .time ()-start_time ,
# Legacy implementation note.
"label_order":[int (x )for x in unique_labels .tolist ()],
"cluster_sizes":cluster_sizes ,
"cluster_centers":cluster_centers 
}
with open (root /f"{base}_{clean_state}_summary.json","w",encoding ="utf-8")as fp :
    json .dump (summary ,fp ,indent =4 )

    # Legacy implementation note.
other_state ="cleaned"if clean_state =="raw"else "raw"
other_path =root /f"{base}_{other_state}_summary.json"
if other_path .exists ():
    other =json .loads (other_path .read_text (encoding ="utf-8"))
    shift ={
    "dataset_id":dataset_id ,
    "delta_k":summary ["best_k"]-other ["best_k"],
    "delta_combined":summary ["combined"]-other ["combined"],
    "rel_shift":abs (summary ["best_k"]-other ["best_k"])/max (other ["best_k"],1 )
    }
    with open (root /f"{base}_param_shift.json","w",encoding ="utf-8")as fp :
        json .dump (shift ,fp ,indent =4 )

        # Legacy implementation note.

        # Legacy implementation note.
        # Legacy implementation note.
df_out =df .loc [used_index ].copy ()
df_out .insert (0 ,"orig_index",df_out .index )# Legacy implementation note.
df_out ["cluster"]=labels_final .astype (int )
df_out .to_csv (root /f"{base}_{clean_state}_clusters.csv",index =False ,encoding ="utf-8")

# Legacy implementation note.
labels_map ={
int (c ):[(int (ix )if isinstance (ix ,(int ,np .integer ))else str (ix ))
for ix in df_out .loc [df_out ["cluster"]==c ,"orig_index"].tolist ()]
for c in unique_labels 
}
with open (root /f"{base}_{clean_state}_labels.json","w",encoding ="utf-8")as fp :
    json .dump (labels_map ,fp ,indent =2 )

    # Legacy implementation note.
index_map =pd .DataFrame ({
"row_pos":np .arange (len (used_index ),dtype =int ),
"orig_index":used_index 
})
index_map .to_csv (root /f"{base}_{clean_state}_index_map.csv",index =False ,encoding ="utf-8")

# Legacy implementation note.
pd .DataFrame (optuna_trials ).to_csv (root /f"{base}_{clean_state}_trials.csv",
index =False ,encoding ="utf-8")
with open (root /f"{base}_{clean_state}_trials.json","w",encoding ="utf-8")as fp :
    json .dump (optuna_trials ,fp ,indent =2 )

    # Legacy implementation note.
def _full_tree_profiles (linkage :str ,metric :str ):
    """Legacy implementation note."""
    hc_full =AgglomerativeClustering (
    n_clusters =None ,
    distance_threshold =0.0 ,
    linkage =linkage ,
    affinity =metric ,
    compute_distances =True 
    )
    hc_full .fit (X_scaled )
    children =hc_full .children_ 
    distances =getattr (hc_full ,"distances_",None )
    n_samples ,n_features =X_scaled .shape 
    n_merges =children .shape [0 ]

    # Legacy implementation note.
    n_nodes =n_samples +n_merges 
    counts =np .zeros (n_nodes ,dtype =np .int64 )
    sums =np .zeros ((n_nodes ,n_features ),dtype =float )
    sumsq =np .zeros (n_nodes ,dtype =float )
    sse_node =np .zeros (n_nodes ,dtype =float )

    counts [:n_samples ]=1 
    sums [:n_samples ]=X_scaled 
    sumsq [:n_samples ]=np .einsum ('ij,ij->i',X_scaled ,X_scaled )

    sse_total_list ,k_list ,dist_list =[],[],[]
    sse_total =0.0 

    for step in range (n_merges ):
        a ,b =int (children [step ,0 ]),int (children [step ,1 ])
        new_id =n_samples +step 

        # Legacy implementation note.
        sse_a ,sse_b =sse_node [a ],sse_node [b ]

        # Legacy implementation note.
        counts [new_id ]=counts [a ]+counts [b ]
        sums [new_id ]=sums [a ]+sums [b ]
        sumsq [new_id ]=sumsq [a ]+sumsq [b ]

        # Legacy implementation note.
        sse_new =float (sumsq [new_id ]-(sums [new_id ]@sums [new_id ])/max (counts [new_id ],1 ))
        sse_node [new_id ]=sse_new 

        # Legacy implementation note.
        sse_total =sse_total -sse_a -sse_b +sse_new 

        # Legacy implementation note.
        sse_total_list .append (sse_total )
        k_list .append (n_samples -(step +1 ))
        dist_list .append (float (distances [step ])if distances is not None else float ("nan"))

    return {
    "children":children ,
    "distances":distances if distances is not None else np .full (len (k_list ),np .nan ),
    "counts":counts ,
    "k_by_step":np .array (k_list ,dtype =int ),
    "sse_by_step":np .array (sse_total_list ,dtype =float ),
    "merge_dist_by_step":np .array (dist_list ,dtype =float )
    }

profiles =_full_tree_profiles (linkage_optuna ,metric_optuna )

# Legacy implementation note.
np .savez (root /f"{base}_{clean_state}_tree.npz",
children =profiles ["children"],
distances =profiles ["distances"],
counts =profiles ["counts"])

# Legacy implementation note.
profile_df =pd .DataFrame ({
"step":np .arange (1 ,len (profiles ["k_by_step"])+1 ,dtype =int ),
"n_clusters":profiles ["k_by_step"],
"merge_distance":profiles ["merge_dist_by_step"],
"sse_total":profiles ["sse_by_step"],
"sse_ratio":profiles ["sse_by_step"]/max (SSE_max ,1e-12 )
})
profile_df .to_csv (root /f"{base}_{clean_state}_tree_profile.csv",
index =False ,encoding ="utf-8")

# Legacy implementation note.
np .save (root /f"{base}_{clean_state}_X_scaled.npy",X_scaled )
with open (root /f"{base}_{clean_state}_columns.json","w",encoding ="utf-8")as fp :
    json .dump ({"columns":X .columns .tolist ()},fp ,indent =2 )

print (f"All files saved in: {root}")
print (f"Program completed in {summary['total_runtime_sec']:.2f} sec")
