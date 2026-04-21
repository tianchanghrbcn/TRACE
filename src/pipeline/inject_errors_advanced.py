#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inject_errors_v2.py  · text“text + text”text
CLI、text。
"""
import argparse ,os ,random ,string 
import numpy as np 
import pandas as pd 

# ---------- CLI ----------
def parse_arguments ():
    p =argparse .ArgumentParser (
    description ="Inject 'anomaly' & 'missing' errors; first column treated as primary key."
    )
    p .add_argument ("--input",required =True ,help ="Clean CSV path")
    p .add_argument ("--output",required =True ,help ="Output directory")
    p .add_argument ("--task_name",required =True ,
    help ="Files: {task_name}_{n}.csv & {task_name}_explanation.txt")
    p .add_argument ("--seed",type =int ,default =None ,help ="Random seed (optional)")
    return p .parse_args ()

    # Legacy implementation note.
def inject_anomaly_and_missing (df ,cols ,a_rate ,m_rate ,rng ):
    """
    text   n_anom = round(N_valid * a_rate)
            n_miss = round(N_valid * m_rate)
    text：
        - text
        - A、M text
    text (n_anom, n_miss)
    """
    # Legacy implementation note.
    candidates =[(r ,c )
    for c in cols 
    for r in df .index 
    if not pd .isna (df .at [r ,c ])]

    N_valid =len (candidates )
    if N_valid ==0 :
        return 0 ,0 

    n_anom =round (N_valid *a_rate )
    n_miss =round (N_valid *m_rate )

    # Legacy implementation note.
    idx_all =rng .choice (N_valid ,size =n_anom +n_miss ,replace =False )
    anom_cells =[candidates [i ]for i in idx_all [:n_anom ]]
    miss_cells =[candidates [i ]for i in idx_all [n_anom :]]

    # Legacy implementation note.
    for (r ,c )in anom_cells :
        df .at [r ,c ]=generate_anomaly_value (df .at [r ,c ],rng )

        # Legacy implementation note.
    for (r ,c )in miss_cells :
        df .at [r ,c ]=np .nan 

    return n_anom ,n_miss 

def generate_anomaly_value (val ,rng ):
    """text 3–6 text；text"""
    if pd .isnull (val ):
        return val 
    try :
        f =float (val )
        return f *rng .uniform (3 ,6 )
    except Exception :
        specs =["#","$","%","??","@@"]
        times =rng .integers (1 ,3 )
        s =str (val )
        for _ in range (times ):
            pos =rng .integers (0 ,len (s )+1 )
            s =s [:pos ]+rng .choice (specs )+s [pos :]
        return s 

        # Legacy implementation note.
def main ():
    args =parse_arguments ()
    if args .seed is not None :
        random .seed (args .seed )
        np .random .seed (args .seed )

    df_clean =pd .read_csv (args .input ).reset_index (drop =True )
    if df_clean .shape [1 ]<2 :
        raise SystemExit ("Dataset text ≥2 text。")

    pk ,other_cols =df_clean .columns [0 ],df_clean .columns [1 :]
    # Legacy implementation note.
    for c in other_cols :
        if pd .api .types .is_integer_dtype (df_clean [c ]):
            df_clean [c ]=df_clean [c ].astype (float )

    os .makedirs (args .output ,exist_ok =True )
    expl_path =os .path .join (args .output ,f"{args.task_name}_explanation.txt")
    rates =[0.00 ,0.05 ,0.10 ,0.15 ]
    rng =np .random .default_rng ()

    explain_lines =[]
    combo_idx =0 
    for a_r in rates :
        for m_r in rates :
            if a_r ==0 and m_r ==0 :
                continue 
            combo_idx +=1 
            df_cor =df_clean .copy ()
            n_a ,n_m =inject_anomaly_and_missing (df_cor ,other_cols ,a_r ,m_r ,rng )

            # Legacy implementation note.
            n_valid =(~df_clean [other_cols ].isna ()).values .sum ()
            r_a =n_a /n_valid 
            r_m =n_m /n_valid 
            r_tot =r_a +r_m 

            fname =f"{args.task_name}_{combo_idx}.csv"
            df_cor .to_csv (os .path .join (args .output ,fname ),index =False )

            explain_lines .append (
            f"{combo_idx:02d} | Anom={a_r:.0%}, Miss={m_r:.0%}  "
            f"→  r_anom={r_a:.2%}, r_miss={r_m:.2%}, r_tot={r_tot:.2%}"
            )
            print (f"[{combo_idx:02d}] {fname} done.")

    with open (expl_path ,"w",encoding ="utf-8")as f :
        f .write ("Error‑injection summary\n\n")
        f .write ("\n".join (explain_lines ))

    print ("\n✅  All corrupted files & explanation saved to:",args .output )

if __name__ =="__main__":
    main ()
