# make_all_figs.py
from pathlib import Path 
from typing import Optional 
import json ,math ,warnings 
import numpy as np 
import pandas as pd 
import matplotlib 
import matplotlib .pyplot as plt 
from matplotlib .font_manager import FontProperties ,fontManager 

# Legacy implementation note.
def pick_cn_font ()->str :
    """Legacy implementation note."""
    preferred =[
    "SimSun","Microsoft YaHei","SimHei",
    "Noto Sans CJK SC","Source Han Sans SC",
    "STSong","FangSong"
    ]
    available ={f .name for f in fontManager .ttflist }
    for name in preferred :
        if name in available :
            return name 
            # Legacy implementation note.
    return "DejaVu Sans"

CN_FONT_NAME =pick_cn_font ()

# Legacy implementation note.
matplotlib .rc ('font',family ='Times New Roman')
matplotlib .rcParams ['axes.unicode_minus']=False # Legacy implementation note.
matplotlib .rcParams ['pdf.fonttype']=42 # Legacy implementation note.
matplotlib .rcParams ['ps.fonttype']=42 

# Legacy implementation note.
TITLE_SIZE =12 
LABEL_SIZE =12 
LEGEND_SIZE =11 
NOTE_SIZE =10 # Legacy implementation note.

cn_font_title =FontProperties (family =CN_FONT_NAME ,size =TITLE_SIZE )
cn_font_label =FontProperties (family =CN_FONT_NAME ,size =LABEL_SIZE )
cn_font_legend =FontProperties (family =CN_FONT_NAME ,size =LEGEND_SIZE )
cn_font_note =FontProperties (family =CN_FONT_NAME ,size =NOTE_SIZE )

warnings .filterwarnings ("ignore",category =UserWarning )

# Legacy implementation note.
ROOT_DIR =Path ("demo_results/clustered_data/HC")
CLEANING_ALGOS =["baran","mode","holoclean"]
DATASET_IDS =[2 ,5 ]# Legacy implementation note.
STEM_PREFIX ="repaired_"# Legacy implementation note.
# ============================================================================

# Legacy implementation note.
def _paths (base_dir :Path ,stem :str ,state :str ):
    p =lambda suf :base_dir /f"{stem}_{state}_{suf}"
    return {
    "summary":p ("summary.json"),
    "clusters":p ("clusters.csv"),
    "x_scaled":p ("X_scaled.npy"),
    "columns":p ("columns.json"),
    "trials_csv":p ("trials.csv"),
    "tree_profile":p ("tree_profile.csv"),
    "tree_npz":p ("tree.npz"),
    }

def _ensure_outdir (d :Path )->Path :
    out =d /"figs"
    out .mkdir (parents =True ,exist_ok =True )
    return out 

def _savefig (fig :plt .Figure ,out_path :Path ,dpi =300 ):
    out_path .parent .mkdir (parents =True ,exist_ok =True )
    fig .tight_layout ()
    fig .savefig (out_path .with_suffix (".png"),dpi =dpi )
    fig .savefig (out_path .with_suffix (".pdf"))
    plt .close (fig )

def _load_json (path :Path ):
    with open (path ,"r",encoding ="utf-8")as f :
        return json .load (f )

def _detect_stems_states (folder :Path ):
    """Legacy implementation note."""
    mapping ={}
    for f in folder .glob ("*_summary.json"):
        name =f .name [:-len ("_summary.json")]
        if "_"not in name :
            continue 
        stem ,state =name .rsplit ("_",1 )
        mapping .setdefault (stem ,[])
        if state not in mapping [stem ]:
            mapping [stem ].append (state )
    return mapping 

def _prefer_state (states ):
    """Legacy implementation note."""
    if "cleaned"in states :
        return "cleaned"
    if "raw"in states :
        return "raw"
    return None 

    # Legacy implementation note.
def _fit_pca_2d (X :np .ndarray ,fit_on :Optional [np .ndarray ]=None ,random_state :int =0 ):
    from sklearn .decomposition import PCA 
    pca =PCA (n_components =2 ,random_state =random_state )
    if fit_on is None :
        return pca ,pca .fit_transform (X )
    else :
        pca .fit (fit_on )
        return pca ,pca .transform (X )

        # Legacy implementation note.
def plot_scatter_with_decision_bg (base_dir :Path ,stem :str ,state :str ,
display_name :str ,fit_on_state :Optional [str ]="cleaned"):
    paths =_paths (base_dir ,stem ,state )
    if not (paths ["x_scaled"].exists ()and paths ["clusters"].exists ()and paths ["summary"].exists ()):
        print (f"[text] text -> {paths}")
        return 
    X =np .load (paths ["x_scaled"])
    dfc =pd .read_csv (paths ["clusters"])
    labels =dfc ["cluster"].to_numpy ()
    sm =_load_json (paths ["summary"])

    # Legacy implementation note.
    sil =sm ["silhouette"]*(1 +sm ["silhouette"])/2 # Legacy implementation note.
    db =1 /(1 +sm ["davies_bouldin"])# Legacy implementation note.

    # Legacy implementation note.
    p_fit =None 
    if fit_on_state is not None :
        fit_paths =_paths (base_dir ,stem ,fit_on_state )
        if fit_paths ["x_scaled"].exists ():
            p_fit =np .load (fit_paths ["x_scaled"])

    _ ,X2d =_fit_pca_2d (X ,fit_on =p_fit )

    uniq =np .unique (labels )
    centers =np .vstack ([X2d [labels ==c ].mean (axis =0 )for c in uniq ])

    x_min ,x_max =X2d [:,0 ].min ()-1 ,X2d [:,0 ].max ()+1 
    y_min ,y_max =X2d [:,1 ].min ()-1 ,X2d [:,1 ].max ()+1 
    xx ,yy =np .meshgrid (np .linspace (x_min ,x_max ,300 ),
    np .linspace (y_min ,y_max ,300 ))
    grid =np .c_ [xx .ravel (),yy .ravel ()]
    assign =((grid [:,None ,:]-centers [None ,:,:])**2 ).sum (-1 ).argmin (1 ).reshape (xx .shape )

    fig =plt .figure (figsize =(4.2 ,3.6 ))
    ax =plt .gca ()
    ax .contourf (xx ,yy ,assign ,alpha =0.15 )# Legacy implementation note.
    for c in uniq :
        idx =labels ==c 
        ax .scatter (X2d [idx ,0 ],X2d [idx ,1 ],s =14 )
    ax .scatter (centers [:,0 ],centers [:,1 ],marker ="X",s =60 ,edgecolors ="k",linewidths =0.6 )

    ax .set_xlabel ("text1",fontproperties =cn_font_label )
    ax .set_ylabel ("text2",fontproperties =cn_font_label )
    # Legacy implementation note.
    ax .set_title (f"{display_name}",fontproperties =cn_font_title )
    ax .text (0.01 ,0.99 ,f"Sil*={sil:.3f}   DB*={db:.3f}",
    transform =ax .transAxes ,va ='top',ha ='left',fontproperties =cn_font_note )

    # Legacy implementation note.
    # ax.legend(frameon=False, fontsize=LEGEND_SIZE, ncol=2, prop=cn_font_legend)

    _savefig (fig ,_ensure_outdir (base_dir )/f"{stem}_{state}_scatter_decision")

    # Legacy implementation note.
def plot_metrics_bars (base_dir :Path ,stem :str ,states :list [str ],display_name :str ):
    rows =[]
    for st in states :
        path =_paths (base_dir ,stem ,st )["summary"]
        if not path .exists ():
            continue 
        sm =_load_json (path )
        rows .append ({
        "state":st ,
        "silhouette":sm ["silhouette"],
        "db_inv":1.0 /(1.0 +sm ["davies_bouldin"]),
        "combined":sm ["combined"],
        })
    if not rows :
        print (f"[text] text summary -> {base_dir}, {stem}")
        return 
    df =pd .DataFrame (rows )
    outdir =_ensure_outdir (base_dir )

    metric_map ={
    "silhouette":"text",
    "db_inv":"1/(1+DB)text",
    "combined":"text",
    }
    for metric_key ,ylabel_cn in metric_map .items ():
        fig =plt .figure (figsize =(3.8 ,3.2 ))
        ax =plt .gca ()
        ax .bar (df ["state"],df [metric_key ])
        ax .set_ylabel (ylabel_cn ,fontproperties =cn_font_label )
        ax .set_xlabel ("text",fontproperties =cn_font_label )
        ax .set_title (f"{display_name}·{ylabel_cn}",fontproperties =cn_font_title )
        _savefig (fig ,outdir /f"{stem}_metrics_{metric_key}")

        # Legacy implementation note.
def plot_hc_process_curves (base_dir :Path ,stem :str ,state :str ,display_name :str ):
    paths =_paths (base_dir ,stem ,state )
    if not (paths ["tree_profile"].exists ()and paths ["summary"].exists ()):
        print (f"[text] text -> {paths}")
        return 
    prof =pd .read_csv (paths ["tree_profile"])
    sm =_load_json (paths ["summary"])
    k_star =sm .get ("best_k",None )

    # A: sse_ratio vs n_clusters
    fig =plt .figure (figsize =(4.2 ,3.2 ))
    ax =plt .gca ()
    ax .semilogy (prof ["n_clusters"],prof ["sse_ratio"])
    ax .invert_xaxis ()
    if k_star is not None :
        ax .axvline (k_star ,linestyle ="--",linewidth =1 )
    ax .set_xlabel ("text k",fontproperties =cn_font_label )
    ax .set_ylabel ("textSSEtext",fontproperties =cn_font_label )
    ax .set_title (f"{display_name}·SSEtext",fontproperties =cn_font_title )
    _savefig (fig ,_ensure_outdir (base_dir )/f"{stem}_{state}_sse_profile")

    # B: merge_distance vs step
    fig =plt .figure (figsize =(4.2 ,3.2 ))
    ax =plt .gca ()
    ax .plot (prof ["step"],prof ["merge_distance"])
    ax .set_xlabel ("text",fontproperties =cn_font_label )
    ax .set_ylabel ("text",fontproperties =cn_font_label )
    ax .set_title (f"{display_name}·text",fontproperties =cn_font_title )
    _savefig (fig ,_ensure_outdir (base_dir )/f"{stem}_{state}_merge_distance")

    # Legacy implementation note.
def plot_dendrogram (base_dir :Path ,stem :str ,state :str ,display_name :str ):
    try :
        from scipy .cluster .hierarchy import dendrogram 
    except Exception :
        print ("[text] text scipytext")
        return 
    paths =_paths (base_dir ,stem ,state )
    if not (paths ["tree_npz"].exists ()and paths ["summary"].exists ()):
        print (f"[text] text -> {paths}")
        return 
    npz =np .load (paths ["tree_npz"])
    children =npz ["children"]
    distances =np .nan_to_num (npz ["distances"].astype (float ),nan =0.0 )
    counts =npz ["counts"].astype (float )
    sm =_load_json (paths ["summary"])
    k_star =sm .get ("best_k",None )

    n_samples =children .shape [0 ]+1 
    Z =np .zeros ((children .shape [0 ],4 ),dtype =float )
    for i ,(a ,b )in enumerate (children ):
        Z [i ,0 ]=a 
        Z [i ,1 ]=b 
        Z [i ,2 ]=distances [i ]if i <len (distances )else 0.0 
        Z [i ,3 ]=counts [n_samples +i ]if (n_samples +i )<len (counts )else 2.0 

    color_th =None 
    if k_star is not None :
        step_cut =max (n_samples -k_star -1 ,0 )
        if 0 <=step_cut <Z .shape [0 ]:
            color_th =Z [step_cut ,2 ]

    fig =plt .figure (figsize =(6.2 ,3.6 ))
    ax =plt .gca ()
    dendrogram (Z ,color_threshold =color_th ,no_labels =True ,count_sort =True ,ax =ax )

    # Legacy implementation note.
    if color_th is not None :
        ax .axhline (color_th ,linestyle ="--",linewidth =1 )

        # Legacy implementation note.
    ax .set_xlabel ("text",fontproperties =cn_font_label ,fontsize =14 )# Legacy implementation note.
    ax .set_ylabel ("text",fontproperties =cn_font_label ,fontsize =14 )# Legacy implementation note.
    ax .set_title (f"{display_name}·textk*={k_star}text",fontproperties =cn_font_title ,fontsize =16 )# Legacy implementation note.

    # Legacy implementation note.
    ax .tick_params (axis ='both',which ='major',labelsize =12 )# Legacy implementation note.

    _savefig (fig ,_ensure_outdir (base_dir )/f"{stem}_{state}_dendrogram")


    # Legacy implementation note.
def plot_trials_overview (base_dir :Path ,stem :str ,state :str ,display_name :str ):
    paths =_paths (base_dir ,stem ,state )
    if not paths ["trials_csv"].exists ():
        print (f"[text] text -> {paths['trials_csv']}")
        return 
    trials =pd .read_csv (paths ["trials_csv"])
    if trials .empty :
        print ("[text] text");return 

        # Legacy implementation note.
    best_idx =trials ["combined_score"].idxmax ()
    best =trials .loc [best_idx ]

    fig =plt .figure (figsize =(4.2 ,3.2 ))
    ax =plt .gca ()
    ax .scatter (trials ["trial_number"],trials ["combined_score"],s =14 )
    ax .scatter ([best ["trial_number"]],[best ["combined_score"]],
    marker ="*",s =100 ,edgecolors ="k")
    ax .set_xlabel ("text",fontproperties =cn_font_label )
    ax .set_ylabel ("text",fontproperties =cn_font_label )
    ax .set_title (f"{display_name}·text",fontproperties =cn_font_title )
    _savefig (fig ,_ensure_outdir (base_dir )/f"{stem}_{state}_trials_trajectory")

    # B) Score vs k
    fig =plt .figure (figsize =(4.2 ,3.2 ))
    ax =plt .gca ()
    ax .scatter (trials ["n_clusters"],trials ["combined_score"],s =14 )
    ax .set_xlabel ("text k",fontproperties =cn_font_label )
    ax .set_ylabel ("text",fontproperties =cn_font_label )
    ax .set_title (f"{display_name}·text-text",fontproperties =cn_font_title )
    _savefig (fig ,_ensure_outdir (base_dir )/f"{stem}_{state}_trials_score_vs_k")

    # Legacy implementation note.
    pivot =(trials .groupby (["linkage","metric"])["combined_score"]
    .max ().reset_index ()
    .pivot (index ="linkage",columns ="metric",values ="combined_score"))
    fig =plt .figure (figsize =(4.6 ,3.4 ))
    ax =plt .gca ()
    im =ax .imshow (pivot .values ,aspect ="auto")
    ax .set_xticks (range (pivot .shape [1 ]));ax .set_xticklabels (list (pivot .columns ))
    ax .set_yticks (range (pivot .shape [0 ]));ax .set_yticklabels (list (pivot .index ))
    plt .colorbar (im ,fraction =0.046 ,pad =0.04 )
    ax .set_title (f"{display_name}·text",fontproperties =cn_font_title )
    # Legacy implementation note.
    for i in range (pivot .shape [0 ]):
        for j in range (pivot .shape [1 ]):
            v =pivot .values [i ,j ]
            if not np .isnan (v ):
                ax .text (j ,i ,f"{v:.3f}",ha ="center",va ="center",fontsize =NOTE_SIZE )
    _savefig (fig ,_ensure_outdir (base_dir )/f"{stem}_{state}_trials_heatmap")

    # Legacy implementation note.
def plot_cluster_sizes (base_dir :Path ,stem :str ,state :str ,display_name :str ):
    paths =_paths (base_dir ,stem ,state )
    if not paths ["summary"].exists ():
        print (f"[text] text summary");return 
    sm =_load_json (paths ["summary"])
    sizes =sm .get ("cluster_sizes",None )
    if sizes is None :
        if not paths ["clusters"].exists ():
            print ("[text] text sizes text clusters.csv");return 
        labels =pd .read_csv (paths ["clusters"])["cluster"].to_numpy ()
        uniq =np .unique (labels )
        sizes =[int (np .sum (labels ==c ))for c in uniq ]
    fig =plt .figure (figsize =(4.2 ,3.2 ))
    ax =plt .gca ()
    ax .bar ([f"text{i}"for i in range (len (sizes ))],sizes )
    ax .set_xlabel ("text",fontproperties =cn_font_label )
    ax .set_ylabel ("text",fontproperties =cn_font_label )
    ax .set_title (f"{display_name}·text",fontproperties =cn_font_title )
    _savefig (fig ,_ensure_outdir (base_dir )/f"{stem}_{state}_cluster_sizes")

    # Legacy implementation note.
def plot_centers_heatmap (base_dir :Path ,stem :str ,state :str ,display_name :str ,topk_features :int =20 ):
    paths =_paths (base_dir ,stem ,state )
    if not (paths ["summary"].exists ()and paths ["columns"].exists ()):
        print (f"[text] text -> {paths}")
        return 
    sm =_load_json (paths ["summary"])
    cols =_load_json (paths ["columns"]).get ("columns",[])
    centers =sm .get ("cluster_centers",None )

    # Legacy implementation note.
    if centers is None :
        if not (paths ["x_scaled"].exists ()and paths ["clusters"].exists ()):
            print ("[text] text centers text x_scaled/clusters");return 
        X =np .load (paths ["x_scaled"])
        labels =pd .read_csv (paths ["clusters"])["cluster"].to_numpy ()
        uniq =np .unique (labels )
        centers =[X [labels ==c ].mean (axis =0 ).tolist ()for c in uniq ]

    centers =np .asarray (centers )# (k, d)
    d =centers .shape [1 ]
    if not cols or len (cols )!=d :
        cols =[f"f{i}"for i in range (d )]

        # Legacy implementation note.
    importance =np .abs (centers ).mean (axis =0 )
    idx =np .argsort (importance )[::-1 ][:min (topk_features ,d )]
    centers_sel =centers [:,idx ]
    cols_sel =[cols [i ]for i in idx ]

    fig =plt .figure (figsize =(min (6.5 ,1.0 +0.35 *len (cols_sel )),3.6 ))
    ax =plt .gca ()
    im =ax .imshow (centers_sel ,aspect ="auto")
    plt .colorbar (im ,fraction =0.046 ,pad =0.04 )
    ax .set_xticks (range (len (cols_sel )));ax .set_xticklabels (cols_sel ,rotation =60 ,ha ="right")
    ax .set_yticks (range (centers_sel .shape [0 ]));ax .set_yticklabels ([f"text{i}"for i in range (centers_sel .shape [0 ])])
    ax .set_title (f"{display_name}·text",fontproperties =cn_font_title )
    _savefig (fig ,_ensure_outdir (base_dir )/f"{stem}_{state}_centers_heatmap")

    # Legacy implementation note.
def plot_sankey_raw_to_cleaned (base_dir :Path ,stem :str ,display_name_left :str ,display_name_right :str ):
    try :
        import plotly .graph_objects as go 
    except Exception :
        print ("[text] text plotlytext Sankeytext")
        return 
    left =_paths (base_dir ,stem ,"raw")["clusters"]
    right =_paths (base_dir ,stem ,"cleaned")["clusters"]
    if not (left .exists ()and right .exists ()):
        print ("[text] Sankeytext raw text cleanedtext");return 

    L =pd .read_csv (left )[["orig_index","cluster"]].rename (columns ={"cluster":"left"})
    R =pd .read_csv (right )[["orig_index","cluster"]].rename (columns ={"cluster":"right"})
    M =L .merge (R ,on ="orig_index",how ="inner")

    left_cats =sorted (M ["left"].unique ())
    right_cats =sorted (M ["right"].unique ())
    lmap ={c :i for i ,c in enumerate (left_cats )}
    rmap ={c :len (left_cats )+j for j ,c in enumerate (right_cats )}

    ct =M .groupby (["left","right"]).size ().reset_index (name ="n")
    sources =[lmap [i ]for i in ct ["left"]]
    targets =[rmap [j ]for j in ct ["right"]]
    values =ct ["n"].tolist ()
    labels =[f"{display_name_left} text{c}"for c in left_cats ]+[f"{display_name_right} text{c}"for c in right_cats ]

    fig =go .Figure (go .Sankey (
    arrangement ="snap",
    node =dict (label =labels ,pad =20 ,thickness =14 ),
    link =dict (source =sources ,target =targets ,value =values )
    ))
    fig .update_layout (title =f"{stem}: {display_name_left} → {display_name_right}")
    outdir =_ensure_outdir (base_dir )
    html_path =outdir /f"{stem}_sankey_raw_to_cleaned.html"
    fig .write_html (str (html_path ))
    try :
        fig .write_image (str (outdir /f"{stem}_sankey_raw_to_cleaned.png"))
        fig .write_image (str (outdir /f"{stem}_sankey_raw_to_cleaned.pdf"))
    except Exception :
        print ("[text] text kaleidotext pip install -U kaleido")
    print (f"[OK] Sankey text{html_path}")

    # Legacy implementation note.
def compare_algorithms_metrics (root_dir :Path ,dataset_id :int ):
    """Legacy implementation note."""
    rows =[]
    for algo in CLEANING_ALGOS :
        base =root_dir /algo /f"clustered_{dataset_id}"
        if not base .exists ():
            continue 
        mapping =_detect_stems_states (base )
        if not mapping :
            continue 
        stem =sorted (mapping .keys ())[0 ]
        st =_prefer_state (mapping [stem ])
        if st is None :
            continue 
        sm =_load_json (_paths (base ,stem ,st )["summary"])
        rows .append ({
        "algo":algo ,
        "state":st ,
        "silhouette":sm ["silhouette"],
        "db_inv":1.0 /(1.0 +sm ["davies_bouldin"]),
        "combined":sm ["combined"],
        })
    if not rows :
        print (f"[text] text {dataset_id} text")
        return 

    cmp_dir =root_dir /f"compare_figs_dataset_{dataset_id}"
    cmp_dir .mkdir (parents =True ,exist_ok =True )
    df =pd .DataFrame (rows )

    metric_map ={
    "silhouette":"text",
    "db_inv":"1/(1+DB)text",
    "combined":"text",
    }

    for metric_key ,ylabel_cn in metric_map .items ():
        fig =plt .figure (figsize =(4.6 ,3.4 ))
        ax =plt .gca ()
        ax .bar (df ["algo"],df [metric_key ])
        ax .set_ylabel (ylabel_cn ,fontproperties =cn_font_label )
        ax .set_xlabel ("text",fontproperties =cn_font_label )
        ax .set_title (f"text {dataset_id}·{ylabel_cn}",fontproperties =cn_font_title )
        _savefig (fig ,cmp_dir /f"dataset{dataset_id}_compare_{metric_key}")

        # Legacy implementation note.
def process_folder_for_algo_dataset (algo :str ,dataset_id :int ):
    base_dir =ROOT_DIR /algo /f"clustered_{dataset_id}"
    if not base_dir .exists ():
        print (f"[text] text{base_dir}")
        return 
    mapping =_detect_stems_states (base_dir )
    if not mapping :
        print (f"[text] text *_summary.jsontext{base_dir}")
        return 

    for stem ,states in mapping .items ():
        display_name =f"{algo}·text{dataset_id}"

        # Legacy implementation note.
        plot_metrics_bars (base_dir ,stem ,sorted (states ),display_name )

        # Legacy implementation note.
        fit_on_state ="cleaned"if "cleaned"in states else None 
        for st in sorted (states ):
            plot_scatter_with_decision_bg (base_dir ,stem ,st ,display_name ,fit_on_state =fit_on_state )
            plot_hc_process_curves (base_dir ,stem ,st ,display_name )
            plot_dendrogram (base_dir ,stem ,st ,display_name )
            plot_trials_overview (base_dir ,stem ,st ,display_name )
            plot_cluster_sizes (base_dir ,stem ,st ,display_name )
            plot_centers_heatmap (base_dir ,stem ,st ,display_name ,topk_features =20 )

            # Legacy implementation note.
        if "raw"in states and "cleaned"in states :
            plot_sankey_raw_to_cleaned (base_dir ,stem ,
            f"{algo}-text",f"{algo}-text")

def main ():
    print (f"[text] text{CN_FONT_NAME}")
    for algo in CLEANING_ALGOS :
        for did in DATASET_IDS :
            process_folder_for_algo_dataset (algo ,did )
    for did in DATASET_IDS :
        compare_algorithms_metrics (ROOT_DIR ,did )
    print ("✅ text figs/ text compare_figs_dataset_{num}/ text")

if __name__ =="__main__":
    main ()
