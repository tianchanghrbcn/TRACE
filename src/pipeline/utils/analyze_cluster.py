#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os 
import json 
import csv 

# Legacy implementation note.
BASE_DIR =os .path .abspath (os .path .join (os .path .dirname (__file__ ),"..","..",".."))
TRAIN_CONFIG_PATH =os .path .join (BASE_DIR ,"src","pipeline","train","comparison.json")
# Legacy implementation note.
OUTPUT_DIR =os .path .join (BASE_DIR ,"results","analysis_results")
os .makedirs (OUTPUT_DIR ,exist_ok =True )

# Legacy implementation note.
TASK_NAMES =["beers","flights","hospital","rayyan"]
CLEANING_METHODS =["mode","bigdansing","boostclean","holoclean","horizon","scared","baran","Unified","GroundTruth"]
# Legacy implementation note.
CLUSTER_METHODS =["KMEANSNF","GMM","HC","KMEANS","KMEANSPPS","DBSCAN"]

# Legacy implementation note.
CSV_FIELDS =[
"task_name","num","dataset_id","error_rate","m","n",
"anomaly","missing",
"cleaning_method","cluster_method","parameters",
"Silhouette Score","Davies-Bouldin Score","Combined Score"
]

def parse_cluster_file (filepath ):

    params =""
    combined =silhouette =davies =None 
    try :
        with open (filepath ,"r",encoding ="utf-8")as f :
            for line in f :
                line =line .strip ()
                if not line :
                    continue 
                if line .startswith ("Best parameters:"):
                # Legacy implementation note.
                    params =line .split (":",1 )[1 ].strip ()
                elif line .startswith ("Final Combined Score:"):
                    combined =float (line .split (":",1 )[1 ].strip ())
                elif line .startswith ("Final Silhouette Score:"):
                    silhouette =float (line .split (":",1 )[1 ].strip ())
                elif line .startswith ("Final Davies-Bouldin Score:"):
                    davies =float (line .split (":",1 )[1 ].strip ())
    except Exception as e :
        print (f"Error reading file {filepath}: {e}")
    return params ,silhouette ,davies ,combined 

def main ():
# Legacy implementation note.
    try :
        with open (TRAIN_CONFIG_PATH ,"r",encoding ="utf-8")as f :
            config_list =json .load (f )
    except Exception as e :
        print (f"Error reading {TRAIN_CONFIG_PATH}: {e}")
        return 

        # Legacy implementation note.
    results_by_task ={task :[]for task in TASK_NAMES }

    for entry in config_list :
        task_name =entry .get ("task_name")
        if task_name not in TASK_NAMES :
            continue # Legacy implementation note.

        num =entry .get ("num")
        dataset_id =entry .get ("dataset_id")
        error_rate =entry .get ("error_rate")
        m_val =entry .get ("m")
        n_val =entry .get ("n")
        # Legacy implementation note.
        details =entry .get ("details","")
        # Legacy implementation note.
        anomaly =missing =fmt =knowledge =None 
        try :
            for part in details .split (","):
                part =part .strip ()
                if part .startswith ("anomaly="):
                    anomaly =float (part .split ("=")[1 ].replace ("%","").strip ())
                elif part .startswith ("missing="):
                    missing =float (part .split ("=")[1 ].replace ("%","").strip ())
        except Exception as e :
            print (f"Error parsing details in entry {entry}: {e}")

            # Legacy implementation note.
        paths =entry .get ("paths",{})
        # Legacy implementation note.
        repaired_paths =paths .get ("repaired_paths",{})

        # Legacy implementation note.
        for cleaning_method ,repaired_csv in repaired_paths .items ():
        # Legacy implementation note.
            if cleaning_method not in CLEANING_METHODS :
                continue 
                # Legacy implementation note.
            for cluster_method in CLUSTER_METHODS :
            # Legacy implementation note.
            # AutoMLClustering\results\clustered_data\{cluster_method}\{cleaning_method}\clustered_{dataset_id}\repaired_{dataset_id}.txt
                cluster_file =os .path .join (
                BASE_DIR ,"results","clustered_data",cluster_method ,
                cleaning_method ,f"clustered_{dataset_id}",
                f"repaired_{dataset_id}.txt"
                )
                if not os .path .exists (cluster_file ):
                # Legacy implementation note.
                # print(f"Cluster result file not found: {cluster_file}")
                    continue 

                    # Legacy implementation note.
                params ,silhouette ,davies ,combined =parse_cluster_file (cluster_file )
                # Legacy implementation note.
                if combined is None or silhouette is None or davies is None :
                # print(f"Incomplete clustering scores in {cluster_file}")
                    continue 

                    # Legacy implementation note.
                row ={
                "task_name":task_name ,
                "num":num ,
                "dataset_id":dataset_id ,
                "error_rate":error_rate ,
                "m":m_val ,
                "n":n_val ,
                "anomaly":anomaly ,
                "missing":missing ,
                "cleaning_method":cleaning_method ,
                "cluster_method":cluster_method ,
                "parameters":params ,
                "Silhouette Score":silhouette ,
                "Davies-Bouldin Score":davies ,
                "Combined Score":combined 
                }
                results_by_task [task_name ].append (row )

                # Legacy implementation note.
    for task in TASK_NAMES :
        output_file =os .path .join (OUTPUT_DIR ,f"{task}_cluster.csv")
        rows =results_by_task .get (task ,[])
        if not rows :
            print (f"No clustering results found for task: {task}")
            continue 
        try :
            with open (output_file ,"w",newline ="",encoding ="utf-8")as csvfile :
                writer =csv .DictWriter (csvfile ,fieldnames =CSV_FIELDS )
                writer .writeheader ()
                for row in rows :
                    writer .writerow (row )
            print (f"Wrote clustering analysis for {task} to {output_file}")
        except Exception as e :
            print (f"Error writing csv for task {task}: {e}")

if __name__ =="__main__":
    main ()
