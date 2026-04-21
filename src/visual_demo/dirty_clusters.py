#!/usr/bin/env python3
# Legacy implementation note.
# --------------------------------------------------------------------
# Legacy implementation note.
# Legacy implementation note.
# Legacy implementation note.
# Legacy implementation note.
# Legacy implementation note.
# --------------------------------------------------------------------

import warnings ,os ,re ,argparse ,math # NEW: math
warnings .filterwarnings ("ignore",message ="Glyph")

import matplotlib 
matplotlib .rcParams .update ({
"font.family":"sans-serif",
"font.sans-serif":["SimHei","Microsoft YaHei","Noto Sans CJK SC"],
"axes.unicode_minus":False ,
"pdf.fonttype":42 
})

import numpy as np 
import pandas as pd 
import matplotlib .pyplot as plt 
from pathlib import Path 

# Legacy implementation note.
DIRTY_DIR ="demo_dirty"
CLEAN_FILE ="clean_withseg.csv"
EXPLAIN_FILE ="rayyan_explanation.txt"
OUTPUT_DIR ="figures/dirty_clusters"

# Legacy implementation note.
SEG_COLOR ={"A":"#1f77b4","B":"#ffbf00","C":"#d62728",
"D":"#17becf","E":"#7f7f7f"}
SEG_ORDER =["A","B","C","D","E"]

# Legacy implementation note.
def parse_explanation (path :str )->dict [int ,str ]:
    pat =re .compile (r"(\d{2}).*?Anom=([\d.]+%).*?Miss=([\d.]+%).*?r_tot=([\d.]+%)")
    out ={}
    if os .path .isfile (path ):
        with open (path ,encoding ="utf-8")as f :
            for line in f :
                m =pat .search (line )
                if m :
                    idx =int (m .group (1 ))
                    out [idx ]=f"text={m.group(2)} text={m.group(3)} text={m.group(4)}"
    return out 

    # Legacy implementation note.
def scatter_clip_arrow (ax ,df ,xmin ,xmax ,ymin ,ymax ,x_thr ,y_thr ):
    dx =0.02 *(xmax -xmin );dy =0.02 *(ymax -ymin )
    x_out ,y_out =xmin -dx ,ymin -dy 

    # Legacy implementation note.
    norm =df .err_type =="normal"
    for seg in SEG_ORDER :
        m =norm &df .segment .str .startswith (seg )
        ax .scatter (df .loc [m ,'age'],df .loc [m ,'income'],
        s =18 ,c =SEG_COLOR [seg ],lw =0 ,alpha =.8 )

        # Legacy implementation note.
    miss_x =(df .err_type =="missing")&df .age .isna ()&df .income .notna ()
    miss_y =(df .err_type =="missing")&df .income .isna ()&df .age .notna ()
    ax .scatter (np .full (miss_x .sum (),x_out ),df .loc [miss_x ,'income'],
    marker ='x',c ='k',s =28 ,lw =1.2 )
    ax .scatter (df .loc [miss_y ,'age'],np .full (miss_y .sum (),y_out ),
    marker ='x',c ='k',s =28 ,lw =1.2 )

    # Legacy implementation note.
    anom =df .err_type =="anomaly"
    for seg in SEG_ORDER :
        m =anom &df .segment .str .startswith (seg )
        for _ ,r in df .loc [m ].iterrows ():
            x =min (r .age ,x_thr *.99 )if not np .isnan (r .age )else x_out 
            y =min (r .income ,y_thr *.99 )if not np .isnan (r .income )else y_out 
            ax .scatter (x ,y ,s =18 ,c =SEG_COLOR [seg ],lw =0 ,alpha =.8 )
            if r .age >x_thr :
                ax .annotate ("",xy =(x ,y ),xytext =(x_thr *.9 ,y ),
                arrowprops =dict (arrowstyle ="->",color =SEG_COLOR [seg ],lw =.8 ))
            if r .income >y_thr :
                ax .annotate ("",xy =(x ,y ),xytext =(x ,y_thr *.9 ),
                arrowprops =dict (arrowstyle ="->",color =SEG_COLOR [seg ],lw =.8 ))

    ax .set_xlim (xmin -dx *1.5 ,x_thr )
    ax .set_ylim (ymin -dy *1.5 ,y_thr )

    # Legacy implementation note.
def main ():
# Legacy implementation note.
    if not os .path .isfile (CLEAN_FILE ):
        raise SystemExit ("❌ text clean_withseg.csv")
    clean =pd .read_csv (CLEAN_FILE )[["ID","age","income","segment"]]
    for c in ("age","income"):
        clean [c ]=pd .to_numeric (clean [c ],errors ="coerce")
    clean .set_index ("ID",inplace =True )

    xmin ,xmax =clean .age .min (),clean .age .max ()
    ymin ,ymax =clean .income .min (),clean .income .max ()
    x_thr =np .percentile (clean .age ,75 )+1.5 *np .subtract (*np .percentile (clean .age ,[75 ,25 ]))
    y_thr =np .percentile (clean .income ,75 )+1.5 *np .subtract (*np .percentile (clean .income ,[75 ,25 ]))

    explain =parse_explanation (Path (DIRTY_DIR )/EXPLAIN_FILE )
    Path (OUTPUT_DIR ).mkdir (parents =True ,exist_ok =True )

    # Legacy implementation note.
    files =[]
    for p in Path (DIRTY_DIR ).glob ("rayyan_*.csv"):
        m =re .search (r"rayyan_(\d+)$",p .stem )
        if m :
            files .append ((int (m .group (1 )),p ))
    files .sort (key =lambda t :t [0 ])# Legacy implementation note.

    if not files :
        raise SystemExit ("❌ text demo_dirty/rayyan_*.csv")

        # Legacy implementation note.
    n_cols =4 
    n_rows =math .ceil (len (files )/n_cols )
    fig_all ,axes =plt .subplots (n_rows ,n_cols ,figsize =(3 *n_cols ,3 *n_rows ))
    fig_all .subplots_adjust (hspace =.42 ,wspace =.17 )
    ax_iter =iter (np .array (axes ).flatten ())

    # Legacy implementation note.
    for idx ,dirty_path in files :
        demo =f"text {idx:02d}"

        dirty =pd .read_csv (dirty_path ).set_index ("ID")
        for c in ("age","income"):
            dirty [c ]=pd .to_numeric (dirty [c ],errors ="coerce")
        df =dirty .join (clean [["age","income","segment"]].rename (
        columns ={"age":"age_true","income":"income_true"}))
        df .reset_index (inplace =True )

        # Legacy implementation note.
        miss =df .age .isna ()|df .income .isna ()
        valid_truth =df [["age_true","income_true"]].notna ().all (axis =1 )
        anom =valid_truth &(~miss )&((df .age !=df .age_true )|(df .income !=df .income_true ))
        df ["err_type"]=np .select ([miss ,anom ],["missing","anomaly"],default ="normal")

        # Legacy implementation note.
        n =len (df )
        miss_rate =(miss ).sum ()/max (n ,1 )
        anom_rate =(anom ).sum ()/max (n ,1 )
        total_rate =((miss |anom )).sum ()/max (n ,1 )
        detail_auto =f"text={anom_rate:.1%} text={miss_rate:.1%} text={total_rate:.1%}"
        detail =explain .get (idx ,detail_auto )

        # Legacy implementation note.
        fig ,ax =plt .subplots (figsize =(5 ,4 ))
        scatter_clip_arrow (ax ,df ,xmin ,xmax ,ymin ,ymax ,x_thr ,y_thr )
        ax .set_xlabel ("text (age)");ax .set_ylabel ("text (income)")
        ax .set_title (f"{demo} | {detail}",fontsize =10 )
        handles =[
        plt .Line2D ([0 ],[0 ],marker ='x',color ='k',ls ='',label ='text',markersize =7 ),
        plt .Line2D ([0 ,1 ],[0 ,0 ],color ='k',lw =1.2 ,marker ='>',markersize =6 ,markevery =[1 ],label ='text')
        ]
        ax .legend (handles =handles ,loc ='upper right',fontsize =8 ,frameon =False )

        plt .tight_layout ()
        single =Path (OUTPUT_DIR )/f"{demo.replace(' ','')}.pdf"
        fig .savefig (single ,dpi =600 );plt .close ()
        print ("✓ text",single )

        # Legacy implementation note.
        ax_small =next (ax_iter )
        scatter_clip_arrow (ax_small ,df ,xmin ,xmax ,ymin ,ymax ,x_thr ,y_thr )
        ax_small .set_xticks ([]);ax_small .set_yticks ([])
        ax_small .set_title (f"{demo} | {detail}",fontsize =7 )

        # Legacy implementation note.
    for ax in ax_iter :
        ax .axis ("off")

    fig_all .suptitle ("text→ text, × text",fontsize =14 )
    fig_all .tight_layout (rect =[0 ,0 ,1 ,0.96 ])
    ov =Path (OUTPUT_DIR )/"dirty_points_overview.pdf"
    fig_all .savefig (ov ,dpi =300 );plt .close ()
    print ("✓ text",ov )

    # ---------------- CLI ----------------
if __name__ =="__main__":
    argparse .ArgumentParser (description ="text Clean text/text").parse_args ()
    main ()
