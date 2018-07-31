# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 03:22:58 2018

@author: Vick
@data: 2018/07/24
@version: 1.0.0
@description: Performance measurement of Single label and Multilabel.
"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def performance_measure(pred_table, true_table):
    n_class = np.size(pred_table, axis=1)
    score = {"AACC":0.0, "MAREC":0.0, "MAPRE":0.0, "MAFM":0.0}
    m_score, cnfsn_mtrx = SClass_measure(pred_table, true_table)
    aacc = 0.0
    marec = 0.0
    mapre = 0.0
    for i in range(n_class):
        aacc += cnfsn_mtrx[i][0]+cnfsn_mtrx[i][3]
        tpfn = cnfsn_mtrx[i][0]+cnfsn_mtrx[i][2]
        tpfp = cnfsn_mtrx[i][0]+cnfsn_mtrx[i][1]
        if (tpfn == 0) or (tpfp == 0):
            if (tpfn+tpfp) != 0:
                marec+=0.0
                mapre+=0.0
            else:
                marec+=1.0
                mapre+=1.0
        else:
            marec += cnfsn_mtrx[i][0]/tpfn
            mapre += cnfsn_mtrx[i][0]/tpfp
    aacc /= len(pred_table)*n_class
    marec /= n_class
    mapre /= n_class
    if mapre+marec == 0.0:
        fm = 0.0
    else:
        fm = 2*marec*mapre/(mapre+marec)
        
#    acc = accuracy_score(true_table, pred_table)
#    prec = precision_score(true_table, pred_table, average='macro')
#    rec = recall_score(true_table, pred_table, average='macro')
#    fm = f1_score(true_table, pred_table, average='macro')
    score["AACC"] = round(aacc, 4)
    score["MAPRE"] = round(mapre, 4)
    score["MAREC"] = round(marec, 4)
    score["MAFM"] = round(fm, 4)
    
    return score


def SClass_measure(pred_table, true_table):
    n_class = np.size(pred_table, axis=1)
    # "REC":0, "PRE":1, "FM":2
    m_score = [0.0, 0.0, 0.0]
    cnfsn_mtrx = []
    score = []
    # "TP":0, "FP":1 "FN":2, "TN":3
    count_mtrx = [0,0,0,0]
    pred = np.equal(pred_table, 1)
    true = np.equal(true_table, 1)
    tp = np.logical_and(pred, true)
    fp = np.logical_and(pred, np.logical_not(true))
    fn = np.logical_and(true, np.logical_not(pred))
    tn = np.logical_and(np.logical_not(pred), np.logical_not(true))
    for i in range(n_class):
        count_mtrx[0] = list(tp[:,i]).count(True)
        count_mtrx[1] = list(fp[:,i]).count(True)
        count_mtrx[2] = list(fn[:,i]).count(True)
        count_mtrx[3] = list(tn[:,i]).count(True)
        cnfsn_mtrx.append(count_mtrx[:])
    for i in range(n_class):
        tpfn = cnfsn_mtrx[i][0]+cnfsn_mtrx[i][2]
        tpfp = cnfsn_mtrx[i][0]+cnfsn_mtrx[i][1]
        if (tpfn == 0) or (tpfp == 0):
            if (tpfn+tpfp) != 0:
                m_score[0] = 0.0
                m_score[1] = 0.0
                m_score[2] = 0.0
            else:
                m_score[0] = 1.0
                m_score[1] = 1.0
                m_score[2] = 1.0
            print("Warning!")
        else:
            # Recall
            m_score[0] = cnfsn_mtrx[i][0]/tpfn
            # Precision
            m_score[1] = cnfsn_mtrx[i][0]/tpfp
            # F-score
            if (m_score[0]+m_score[1]) != 0.0:
                m_score[2] = (2*m_score[0]*m_score[1])/(m_score[0]+m_score[1])
            else:
                m_score[2] = 0.0
        score.append(m_score[:])
    return score, cnfsn_mtrx


def performance_measureML(pred_table, true_table):
    n_class = np.size(pred_table, axis=1)
    score = {"EMR":0.0, "LFM":0.0, "HL":0.0}
    
    emr = (pred_table == true_table)
    score_emr = 0
    score_hl = 0
    score_lfm = 0
    for i in emr:
        if (False not in i):
            score_emr += 1
        score_hl += list(i).count(False)
    for j in range(len(pred_table)):
        s = (pred_table[j]+true_table[j]).sum()
        if s != 0.0:
            score_lfm += 2*(pred_table[j]*true_table[j]).sum()/s
        else:
            score_lfm += 1.0

    score["EMR"] = score_emr/len(pred_table)
    score["LFM"] = score_lfm/len(pred_table)
    score["HL"] = score_hl/n_class/len(pred_table)    
    
    return score

def performance_Fmeasure(pred_table, true_table):
    score = {"ACC":0.0, "PRE":0.0, "REC":0.0, "FM":0.0}

    acc = accuracy_score(true_table, pred_table)
    prec = precision_score(true_table, pred_table, average='macro')
    rec = recall_score(true_table, pred_table, average='macro')
    fm = f1_score(true_table, pred_table, average='macro')
    
    score["ACC"] = acc
    score["PRE"] = prec
    score["REC"] = rec
    score["FM"] = fm
    
    return score