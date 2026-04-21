"""
merge_to_summary.py
-------------------
text <dataset>_cleaning.csv text <dataset>_cluster.csv text <dataset>_summary.xlsxtext

text:
    pandas>=1.3
    openpyxl   # Legacy implementation note. pandas.to_excel text

text:
    python merge_to_summary.py      # Legacy implementation note.
    python merge_to_summary.py beers  # Legacy implementation note.
"""
from pathlib import Path 
import sys 
import pandas as pd 

BASE_DIR =Path ("../../../results/analysis_results")
DATASETS =["beers","hospital","flights","rayyan"]

# Legacy implementation note.
KEY_COLS =[
"task_name","num","dataset_id","error_rate",
"m","n","anomaly","missing","cleaning_method"
]

# Legacy implementation note.
FINAL_COL_ORDER =[
"task_name","num","dataset_id","error_rate","m","n",
"anomaly","missing","cleaning_method",
"precision","recall","F1","EDR",
"cluster_method","parameters",
"Silhouette Score","Davies-Bouldin Score","Combined Score",
"Sil_relative","DB_relative","Comb_relative",
]

def build_summary (ds :str )->pd .DataFrame :
    """text summary DataFrame text xlsxtext"""
    cleaning_path =BASE_DIR /f"{ds}_cleaning.csv"
    cluster_path =BASE_DIR /f"{ds}_cluster.csv"
    out_path =BASE_DIR /f"{ds}_summary.xlsx"

    # Legacy implementation note.
    clean_df =pd .read_csv (cleaning_path )
    cluster_df =pd .read_csv (cluster_path )

    # Legacy implementation note.
    merged =pd .merge (clean_df ,cluster_df ,on =KEY_COLS ,how ="inner")

    # Legacy implementation note.
    gt =(
    merged [merged ["cleaning_method"]=="GroundTruth"]
    .loc [:,["task_name","num","dataset_id","cluster_method",
    "Silhouette Score","Davies-Bouldin Score","Combined Score"]]
    .rename (columns ={
    "Silhouette Score":"Sil_gt",
    "Davies-Bouldin Score":"DB_gt",
    "Combined Score":"Comb_gt",
    })
    )

    # Legacy implementation note.
    merged =merged .merge (
    gt ,
    on =["task_name","num","dataset_id","cluster_method"],
    how ="left",
    )

    # Legacy implementation note.
    merged ["Sil_relative"]=merged ["Silhouette Score"]/merged ["Sil_gt"]
    merged ["DB_relative"]=merged ["DB_gt"]/merged ["Davies-Bouldin Score"]
    merged ["Comb_relative"]=merged ["Combined Score"]/merged ["Comb_gt"]

    # Legacy implementation note.
    summary =merged [FINAL_COL_ORDER ].copy ()

    # Legacy implementation note.
    summary .to_excel (out_path ,index =False )
    print (f"[✓] {ds}: text {out_path.relative_to(BASE_DIR.parent.parent)}")

    return summary 

def main ():
    targets =DATASETS if len (sys .argv )==1 else [sys .argv [1 ]]
    for ds in targets :
        if ds not in DATASETS :
            print (f"[!] text: {ds}")
            continue 
        build_summary (ds )

if __name__ =="__main__":
    main ()
