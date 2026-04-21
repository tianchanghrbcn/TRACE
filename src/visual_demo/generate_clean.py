#!/usr/bin/env python3
# Legacy implementation note.
# ------------------------------------------------------------------
# Legacy implementation note.
# Legacy implementation note.
# ------------------------------------------------------------------

import numpy as np 
import pandas as pd 
import matplotlib .pyplot as plt 

# Legacy implementation note.
SEED =42 
N_CLUSTERS =5 
N_PER_CLUSTER =60 
OUT_WITH_SEG ="clean_withseg.csv"
OUT_NO_SEG ="clean_noseg.csv"
OUT_FIG_PDF ="reference_clusters.pdf"

# Legacy implementation note.
SEG_COLOR ={"A":"#1f77b4",# Legacy implementation note.
"B":"#ffbf00",# Legacy implementation note.
"C":"#d62728",# Legacy implementation note.
"D":"#17becf",# Legacy implementation note.
"E":"#7f7f7f"}# Legacy implementation note.
SEG_ORDER =["A","B","C","D","E"]

# Legacy implementation note.
DIST ={
#     μ_age, σ_age,  min, max     μ_inc, σ_inc,  min,  max
"A":{"age":(22 ,4 ,16 ,30 ),
"inc":(22_000 ,6_000 ,8_000 ,32_000 )},# Legacy implementation note.

"B":{"age":(31 ,4 ,23 ,39 ),# μ_age +1
"inc":(38_000 ,6_000 ,18_000 ,50_000 )},# μ_inc +3k, σ↓2k

"C":{"age":(40 ,4 ,32 ,48 ),# μ_age +1
"inc":(61_000 ,6_000 ,40_000 ,78_000 )},# μ_inc +4k, σ↓2k

"D":{"age":(50 ,4 ,42 ,58 ),
"inc":(86_000 ,8_000 ,60_000 ,108_000 )},# σ_inc ↓2k

"E":{"age":(63 ,3 ,56 ,70 ),# σ_age ↓1
"inc":(54_000 ,6_000 ,33_000 ,72_000 )},# σ_inc ↓2k, μ_inc +2k
}


# Legacy implementation note.
def _gen_cluster (label :str ,n :int )->pd .DataFrame :
    cfg =DIST [label ]
    age =np .random .normal (*cfg ["age"][:2 ],n ).clip (*cfg ["age"][2 :]).round ().astype (int )
    inc =np .random .normal (*cfg ["inc"][:2 ],n ).clip (*cfg ["inc"][2 :]).round (-2 ).astype (int )
    return pd .DataFrame ({"age":age ,"income":inc ,"segment":label })

    # Legacy implementation note.
def main ()->None :
    np .random .seed (SEED )

    labels_full =list (DIST )[:N_CLUSTERS ]
    df =pd .concat ([_gen_cluster (lab ,N_PER_CLUSTER )for lab in labels_full ],
    ignore_index =True )

    # Legacy implementation note.
    df =df .sample (frac =1 ,random_state =SEED ).reset_index (drop =True )
    df .insert (0 ,"ID",np .arange (1 ,len (df )+1 ))

    # Legacy implementation note.
    df .to_csv (OUT_WITH_SEG ,index =False ,encoding ="utf-8-sig")
    df .drop (columns =["segment"]).to_csv (OUT_NO_SEG ,index =False ,
    encoding ="utf-8-sig")
    print (f"[OK] text {len(df)} text → {OUT_WITH_SEG} / {OUT_NO_SEG}")

    # Legacy implementation note.
    plt .rcParams ["font.sans-serif"]=["SimHei"]# Legacy implementation note.
    plt .rcParams ["axes.unicode_minus"]=False 
    legend_map ={lab :SEG_ORDER [i ]for i ,lab in enumerate (labels_full )}

    plt .figure (figsize =(6 ,4 ))
    for idx ,lab in enumerate (labels_full ):
        seg_letter =legend_map [lab ]# A/B/C/D/E
        color =SEG_COLOR [seg_letter ]
        mask =df ["segment"]==lab 
        plt .scatter (df .loc [mask ,"age"],df .loc [mask ,"income"],
        s =30 ,color =color ,label =seg_letter )

    plt .xlabel ("text")
    plt .ylabel ("text")
    plt .title ("text-text")
    plt .legend (loc ="best",fontsize =9 ,title ="text")
    plt .tight_layout ()
    plt .savefig (OUT_FIG_PDF ,dpi =300 )
    plt .close ()
    print (f"[OK] text: {OUT_FIG_PDF}")

if __name__ =="__main__":
    main ()
