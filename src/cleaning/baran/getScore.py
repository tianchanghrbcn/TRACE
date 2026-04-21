import os 
import sys 
import pandas as pd 
from sklearn .metrics import mean_squared_error ,jaccard_score 
import numpy as np 


def calculate_all_metrics (clean ,dirty ,cleaned ,attributes ,output_path ,task_name ,index_attribute ='index',calculate_precision_recall =True ,
calculate_edr =True ,calculate_hybrid =True ,calculate_r_edr =True ,mse_attributes =[]):
    """
    textEDRtexttext R-EDRtext

    :param clean: text DataFrame
    :param dirty: text DataFrame
    :param cleaned: text DataFrame
    :param attributes: text
    :param output_path: text
    :param task_name: text
    :param calculate_precision_recall: text
    :param calculate_edr: texttextEDRtext
    :param calculate_hybrid: text
    :param calculate_r_edr: texttextR-EDRtext
    :return: text
    """

    results ={}

    # Legacy implementation note.
    if calculate_precision_recall :
        accuracy ,recall =calculate_accuracy_and_recall (clean ,dirty ,cleaned ,attributes ,output_path ,task_name ,index_attribute =index_attribute )
        results ['accuracy']=accuracy 
        results ['recall']=recall 
        f1_score =calF1 (accuracy ,recall )
        results ['f1_score']=f1_score 
        print (f"text: {accuracy}, text: {recall}, F1text: {f1_score}")
        print ("="*40 )

        # Legacy implementation note.
    if calculate_edr :
        edr =get_edr (clean ,dirty ,cleaned ,attributes ,output_path ,task_name ,index_attribute =index_attribute )
        results ['edr']=edr 
        print (f"text (EDR): {edr}")
        print ("="*40 )

        # Legacy implementation note.
    if calculate_hybrid :
        hybrid_distance =get_hybrid_distance (clean ,cleaned ,attributes ,output_path ,task_name ,index_attribute =index_attribute ,mse_attributes =mse_attributes )
        results ['hybrid_distance']=hybrid_distance 
        print (f"text (Hybrid Distance): {hybrid_distance}")
        print ("="*40 )

        # Legacy implementation note.
    if calculate_r_edr :
        r_edr =get_record_based_edr (clean ,dirty ,cleaned ,output_path ,task_name ,index_attribute =index_attribute )
        results ['r_edr']=r_edr 
        print (f"text (R-EDR): {r_edr}")
        print ("="*40 )

    return results 
def normalize_value (value ):
    """
    texttexttext
    :param value: text
    :return: text
    """
    try :
    # Legacy implementation note.
        float_value =float (value )
        if float_value .is_integer ():
            return str (int (float_value ))# Legacy implementation note.
        else :
            return str (float_value )
    except ValueError :
    # Legacy implementation note.
        return str (value )


def default_distance_func (value1 ,value2 ):
    """
    text1text0text
    """
    return (value1 !=value2 ).sum ()

def record_based_distance_func (row1 ,row2 ):
    """
    text1text0text
    """
    for val1 ,val2 in zip (row1 ,row2 ):
        if val1 !=val2 :
            return 1 # Legacy implementation note.
    return 0 # Legacy implementation note.
def calF1 (precision ,recall ):
    """
    textF1text

    :param precision: text
    :param recall: text
    :return: F1text
    """
    return 2 *precision *recall /(precision +recall +1e-10 )


def calculate_accuracy_and_recall (clean ,dirty ,cleaned ,attributes ,output_path ,task_name ,index_attribute ='index'):
    """
    textCSV texttext

    :param clean: text DataFrame
    :param dirty: text DataFrame
    :param cleaned: text DataFrame
    :param attributes: text
    :param output_path: text
    :param task_name: texttexttext
    :param index_attribute: text
    :return: text
    """

    os .makedirs (output_path ,exist_ok =True )

    # Legacy implementation note.
    out_path =os .path .join (output_path ,f"{task_name}_evaluation.txt")

    # Legacy implementation note.
    clean_dirty_diff_path =os .path .join (output_path ,f"{task_name}_clean_vs_dirty.csv")
    dirty_cleaned_diff_path =os .path .join (output_path ,f"{task_name}_dirty_vs_cleaned.csv")
    clean_cleaned_diff_path =os .path .join (output_path ,f"{task_name}_clean_vs_cleaned.csv")

    # Legacy implementation note.
    original_stdout =sys .stdout 

    # Legacy implementation note.
    clean =clean .set_index (index_attribute ,drop =False )
    dirty =dirty .set_index (index_attribute ,drop =False )
    cleaned =cleaned .set_index (index_attribute ,drop =False )

    # Legacy implementation note.
    with open (out_path ,'w',encoding ='utf-8')as f :
        sys .stdout =f # Legacy implementation note.

        total_true_positives =0 
        total_false_positives =0 
        total_true_negatives =0 

        # Legacy implementation note.
        clean_dirty_diff =pd .DataFrame (columns =['Attribute','Index','Clean Value','Dirty Value'])
        dirty_cleaned_diff =pd .DataFrame (columns =['Attribute','Index','Dirty Value','Cleaned Value'])
        clean_cleaned_diff =pd .DataFrame (columns =['Attribute','Index','Clean Value','Cleaned Value'])

        for attribute in attributes :
        # Legacy implementation note.
            clean_values =clean [attribute ].apply (normalize_value )
            dirty_values =dirty [attribute ].apply (normalize_value )
            cleaned_values =cleaned [attribute ].apply (normalize_value )

            # Legacy implementation note.
            common_indices =clean_values .index .intersection (cleaned_values .index ).intersection (dirty_values .index )
            clean_values =clean_values .loc [common_indices ]
            dirty_values =dirty_values .loc [common_indices ]
            cleaned_values =cleaned_values .loc [common_indices ]

            # Legacy implementation note.
            true_positives =((cleaned_values ==clean_values )&(dirty_values !=cleaned_values )).sum ()
            # Legacy implementation note.
            false_positives =((cleaned_values !=clean_values )&(dirty_values !=cleaned_values )).sum ()
            # Legacy implementation note.
            true_negatives =(dirty_values !=clean_values ).sum ()

            # Legacy implementation note.
            mismatched_indices =dirty_values [dirty_values !=clean_values ].index 
            clean_dirty_diff =pd .concat ([clean_dirty_diff ,pd .DataFrame ({
            'Attribute':attribute ,
            'Index':mismatched_indices ,
            'Clean Value':clean_values .loc [mismatched_indices ],
            'Dirty Value':dirty_values .loc [mismatched_indices ]
            })])

            # Legacy implementation note.
            cleaned_indices =cleaned_values [cleaned_values !=dirty_values ].index 
            dirty_cleaned_diff =pd .concat ([dirty_cleaned_diff ,pd .DataFrame ({
            'Attribute':attribute ,
            'Index':cleaned_indices ,
            'Dirty Value':dirty_values .loc [cleaned_indices ],
            'Cleaned Value':cleaned_values .loc [cleaned_indices ]
            })])

            # Legacy implementation note.
            clean_cleaned_indices =cleaned_values [cleaned_values !=clean_values ].index 
            clean_cleaned_diff =pd .concat ([clean_cleaned_diff ,pd .DataFrame ({
            'Attribute':attribute ,
            'Index':clean_cleaned_indices ,
            'Clean Value':clean_values .loc [clean_cleaned_indices ],
            'Cleaned Value':cleaned_values .loc [clean_cleaned_indices ]
            })])

            total_true_positives +=true_positives 
            total_false_positives +=false_positives 
            total_true_negatives +=true_negatives 
            print ("Attribute:",attribute ,"text:",true_positives ,"text:",false_positives ,
            "text:",true_negatives )
            print ("="*40 )

            # Legacy implementation note.
        accuracy =total_true_positives /(total_true_positives +total_false_positives )
        # Legacy implementation note.
        recall =total_true_positives /total_true_negatives 

        # Legacy implementation note.
        print (f"text: {accuracy}")
        print (f"text: {recall}")

        # Legacy implementation note.
    sys .stdout =original_stdout 

    # Legacy implementation note.
    clean_dirty_diff .to_csv (clean_dirty_diff_path ,index =False )
    dirty_cleaned_diff .to_csv (dirty_cleaned_diff_path ,index =False )
    clean_cleaned_diff .to_csv (clean_cleaned_diff_path ,index =False )

    print (f"text:\n{clean_dirty_diff_path}\n{dirty_cleaned_diff_path}\n{clean_cleaned_diff_path}")

    return accuracy ,recall 


def get_edr (clean ,dirty ,cleaned ,attributes ,output_path ,task_name ,index_attribute ='index',distance_func =default_distance_func ):
    """
    text (EDR)texttexttext

    :param clean: text DataFrame
    :param dirty: text DataFrame
    :param cleaned: text DataFrame
    :param attributes: text
    :param output_path: text
    :param task_name: texttexttext
    :param index_attribute: text
    :param distance_func: text1texttext0
    :return: text (EDR)
    """

    # Legacy implementation note.
    os .makedirs (output_path ,exist_ok =True )

    # Legacy implementation note.
    out_path =os .path .join (output_path ,f"{task_name}_edr_evaluation.txt")

    # Legacy implementation note.
    original_stdout =sys .stdout 

    # Legacy implementation note.
    clean =clean .set_index (index_attribute ,drop =False )
    dirty =dirty .set_index (index_attribute ,drop =False )
    cleaned =cleaned .set_index (index_attribute ,drop =False )

    # Legacy implementation note.
    with open (out_path ,'w',encoding ='utf-8')as f :
        sys .stdout =f # Legacy implementation note.

        total_distance_dirty_to_clean =0 
        total_distance_repaired_to_clean =0 

        for attribute in attributes :
        # Legacy implementation note.
            clean_values =clean [attribute ].apply (normalize_value )
            dirty_values =dirty [attribute ].apply (normalize_value )
            cleaned_values =cleaned [attribute ].apply (normalize_value )

            # Legacy implementation note.
            common_indices =clean_values .index .intersection (cleaned_values .index ).intersection (dirty_values .index )
            clean_values =clean_values .loc [common_indices ]
            dirty_values =dirty_values .loc [common_indices ]
            cleaned_values =cleaned_values .loc [common_indices ]

            # Legacy implementation note.
            distance_dirty_to_clean =distance_func (dirty_values ,clean_values )
            # Legacy implementation note.
            distance_repaired_to_clean =distance_func (cleaned_values ,clean_values )

            total_distance_dirty_to_clean +=distance_dirty_to_clean 
            total_distance_repaired_to_clean +=distance_repaired_to_clean 

            # Legacy implementation note.
            print (f"Attribute: {attribute}")
            print (f"Distance (Dirty to Clean): {distance_dirty_to_clean}")
            print (f"Distance (Repaired to Clean): {distance_repaired_to_clean}")
            print ("="*40 )

            # Legacy implementation note.
        if total_distance_dirty_to_clean ==0 :
            edr =0 
        else :
            edr =(total_distance_dirty_to_clean -total_distance_repaired_to_clean )/total_distance_dirty_to_clean 

            # Legacy implementation note.
        print (f"text: {total_distance_dirty_to_clean}")
        print (f"text: {total_distance_repaired_to_clean}")
        print (f"text (EDR): {edr}")

        # Legacy implementation note.
    sys .stdout =original_stdout 

    print (f"EDR text: {out_path}")

    return edr 

def get_hybrid_distance (clean ,cleaned ,attributes ,output_path ,task_name ,index_attribute ='index',mse_attributes =[],w1 =0.5 ,w2 =0.5 ):
    """
    texttexttextMSEtextJaccardtext:param clean: text DataFrame
    :param cleaned: text DataFrame
    :param attributes: text
    :param output_path: text
    :param task_name: texttexttext
    :param index_attribute: text
    :param w1: MSEtext
    :param w2: Jaccardtext
    :param mse_attributes: textMSEtext
    :return: text
    """

    # Legacy implementation note.
    os .makedirs (output_path ,exist_ok =True )

    # Legacy implementation note.
    out_path =os .path .join (output_path ,f"{task_name}_hybrid_distance_evaluation.txt")

    # Legacy implementation note.
    original_stdout =sys .stdout 

    # Legacy implementation note.
    clean =clean .set_index (index_attribute ,drop =False )
    cleaned =cleaned .set_index (index_attribute ,drop =False )

    # Legacy implementation note.
    with open (out_path ,'w',encoding ='utf-8')as f :
        sys .stdout =f # Legacy implementation note.

        total_mse =0 
        total_jaccard =0 
        attribute_count =0 

        for attribute in attributes :
        # Legacy implementation note.
            clean_values =clean [attribute ].apply (normalize_value )
            cleaned_values =cleaned [attribute ].apply (normalize_value )

            # Legacy implementation note.
            clean_values =clean_values .replace ('empty',np .nan )
            cleaned_values =cleaned_values .replace ('empty',np .nan )

            # Legacy implementation note.
            if attribute in mse_attributes :
            # Legacy implementation note.
                try :
                    mse =mean_squared_error (clean_values .dropna ().astype (float ),cleaned_values .dropna ().astype (float ))
                except ValueError :
                    print (f"text {attribute} texttext")
                    mse =np .nan # Legacy implementation note.
            else :
                mse =np .nan 

                # Legacy implementation note.
            try :
            # Legacy implementation note.
                common_indices =clean_values .dropna ().index .intersection (cleaned_values .dropna ().index )
                jaccard =1 -jaccard_score (clean_values .loc [common_indices ],cleaned_values .loc [common_indices ],average ='macro')
            except ValueError :
                print (f"textJaccardtexttexttext {attribute} text")
                jaccard =np .nan # Legacy implementation note.

                # Legacy implementation note.
            if not np .isnan (mse )and not np .isnan (jaccard ):
                total_mse +=mse 
                total_jaccard +=jaccard 
                attribute_count +=1 
            elif not np .isnan (mse )and np .isnan (jaccard ):
                total_mse +=mse 
                attribute_count +=1 
            elif np .isnan (mse )and not np .isnan (jaccard ):
                total_jaccard +=jaccard 
                attribute_count +=1 
            else :
                print (f"texttexttext {attribute} text")

            print (f"Attribute: {attribute}, MSE: {mse}, Jaccard: {jaccard}")

        if attribute_count ==0 :
            return None 

            # Legacy implementation note.
        avg_mse =total_mse /attribute_count 
        avg_jaccard =total_jaccard /attribute_count 

        hybrid_distance =w1 *avg_mse +w2 *avg_jaccard 

        print (f"text: {hybrid_distance}")

        # Legacy implementation note.
    sys .stdout =original_stdout 

    print (f"text: {out_path}")

    return hybrid_distance 

def get_record_based_edr (clean ,dirty ,cleaned ,output_path ,task_name ,index_attribute ='index'):
    """
    text (R-EDR)texttext R-EDR texttext

    :param clean: text DataFrame
    :param dirty: text DataFrame
    :param cleaned: text DataFrame
    :param output_path: text
    :param task_name: texttexttext
    :param index_attribute: text
    :return: text (R-EDR)
    """

    # Legacy implementation note.
    os .makedirs (output_path ,exist_ok =True )

    # Legacy implementation note.
    out_path =os .path .join (output_path ,f"{task_name}_record_based_edr_evaluation.txt")

    # Legacy implementation note.
    original_stdout =sys .stdout 

    # Legacy implementation note.
    clean =clean .set_index (index_attribute ,drop =False )
    dirty =dirty .set_index (index_attribute ,drop =False )
    cleaned =cleaned .set_index (index_attribute ,drop =False )

    total_distance_dirty_to_clean =0 
    total_distance_repaired_to_clean =0 

    # Legacy implementation note.
    with open (out_path ,'w',encoding ='utf-8')as f :
        sys .stdout =f # Legacy implementation note.

        # Legacy implementation note.
        for idx in clean .index :
            clean_row =clean .loc [idx ].apply (normalize_value )
            dirty_row =dirty .loc [idx ].apply (normalize_value )
            cleaned_row =cleaned .loc [idx ].apply (normalize_value )

            # Legacy implementation note.
            distance_dirty_to_clean =record_based_distance_func (dirty_row ,clean_row )
            # Legacy implementation note.
            distance_repaired_to_clean =record_based_distance_func (cleaned_row ,clean_row )

            total_distance_dirty_to_clean +=distance_dirty_to_clean 
            total_distance_repaired_to_clean +=distance_repaired_to_clean 

            # Legacy implementation note.
            print (f"Record {idx}")
            print (f"Distance (Dirty to Clean): {distance_dirty_to_clean}")
            print (f"Distance (Repaired to Clean): {distance_repaired_to_clean}")
            print ("="*40 )

            # Legacy implementation note.
        if total_distance_dirty_to_clean ==0 :
            r_edr =0 
        else :
            r_edr =(total_distance_dirty_to_clean -total_distance_repaired_to_clean )/total_distance_dirty_to_clean 

            # Legacy implementation note.
        print (f"text: {total_distance_dirty_to_clean}")
        print (f"text: {total_distance_repaired_to_clean}")
        print (f"text (R-EDR): {r_edr}")

        # Legacy implementation note.
    sys .stdout =original_stdout 

    print (f"R-EDR text: {out_path}")

    return r_edr 

def test_calculate_all_metrics ():
# Legacy implementation note.
    data ={
    'index1':[1 ,2 ,3 ,4 ,5 ],
    'Attribute1':[1 ,2 ,3 ,4 ,5 ],
    'Attribute2':['A','B','C','D','E'],
    'Attribute3':[1.1 ,2.2 ,3.3 ,4.4 ,5.5 ]
    }

    # Legacy implementation note.
    clean_df =pd .DataFrame (data )

    # Legacy implementation note.
    dirty_data ={
    'index1':[1 ,2 ,3 ,4 ,5 ],
    'Attribute1':[1 ,9 ,3 ,4 ,5 ],# Legacy implementation note.
    'Attribute2':['A','B','X','D','E'],# Legacy implementation note.
    'Attribute3':[1.1 ,2.2 ,3.3 ,4.4 ,5.5 ]# Legacy implementation note.
    }
    dirty_df =pd .DataFrame (dirty_data )

    # Legacy implementation note.
    cleaned_data ={
    'index1':[1 ,2 ,3 ,4 ,5 ],
    'Attribute1':[1 ,9 ,3 ,4 ,5 ],# Legacy implementation note.
    'Attribute2':['A','X','C','D','E'],# Legacy implementation note.
    'Attribute3':[1.1 ,2.2 ,3.3 ,4.4 ,5.7 ]# Legacy implementation note.
    }
    cleaned_df =pd .DataFrame (cleaned_data )

    # Legacy implementation note.
    attributes =['Attribute1','Attribute2','Attribute3']

    # Legacy implementation note.
    output_path ='./temp_test_output'
    task_name ='test_task'

    # Legacy implementation note.
    results =calculate_all_metrics (clean_df ,dirty_df ,cleaned_df ,attributes ,output_path ,task_name ,index_attribute ='index1',mse_attributes =['Attribute3'])

    # Legacy implementation note.
    print ("text:")
    print (f"Accuracy: {results.get('accuracy')}")
    print (f"Recall: {results.get('recall')}")
    print (f"F1 Score: {results.get('f1_score')}")
    print (f"EDR: {results.get('edr')}")
    print (f"Hybrid Distance: {results.get('hybrid_distance')}")
    print (f"R-EDR: {results.get('r_edr')}")

    # Legacy implementation note.
    # assert results['accuracy'] > 0, "Accuracy should be greater than 0"
    # assert results['recall'] > 0, "Recall should be greater than 0"
    # assert results['f1_score'] > 0, "F1 score should be greater than 0"
    # assert results['edr'] > 0, "EDR should be greater than 0"
    # assert results['r_edr'] > 0, "R-EDR should be greater than 0"

    print ("texttext")

if __name__ =="__main__":
# Legacy implementation note.
    test_calculate_all_metrics ()