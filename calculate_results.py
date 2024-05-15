# Code is referenced form the paper "Sewer-ML:A Multi-Label Sewer Defect Classification Dataset and Benchmark"
# you can find the original code here: https://bitbucket.org/aauvap/sewer-ml/src/master/

import csv
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
import argparse
import pandas as pd
from lib.utils.metrics_sewerML import evaluation


LabelWeightDict = {"RB":1.00,"OB":0.5518,"PF":0.2896,"DE":0.1622,"FS":0.6419,"IS":0.1847,"RO":0.3559,"IN":0.3131,"AF":0.0811,"BE":0.2275,"FO":0.2477,"GR":0.0901,"PH":0.4167,"PB":0.4167,"OS":0.9009,"OP":0.3829,"OK":0.4396}
Labels = list(LabelWeightDict.keys())
LabelWeights = list(LabelWeightDict.values())

def calculateResults(args):
    scorePath = args["score_path"]
    targetPath = args["gt_path"]

    outputPath = os.path.join(scorePath, 'result')
    scorePath = outputPath
    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)

    split = args["split"]

    targetSplitpath = os.path.join(targetPath, args["val_csv_name"].format(split))
    #targetSplitpath = os.path.join(targetPath, "SewerML_Val.csv".format(split))
    targetsDf = pd.read_csv(targetSplitpath, sep=",")
    targetsname = targetsDf.sort_values(by=["Filename"]).reset_index(drop=True)
    targets = targetsDf[Labels].values

    print(scorePath)
    for subdir, dirs, files in os.walk(scorePath):
        print(files)
        for scoreFile in files:
            scoresDf = pd.read_csv(os.path.join(subdir, scoreFile), sep=",")
            scores = scoresDf[Labels].values
            new, main, auxillary = evaluation(scores, targets, LabelWeights)
            outputName = "{}_{}".format(split, scoreFile.split('_')[0])

            with open(os.path.join(outputPath,'{}.json'.format(outputName)), 'w') as fp:
                json.dump({"Labels": Labels, "LabelWeights": LabelWeights, "New": new, "Main": main, "Auxillary": auxillary}, fp)

            newString = "{:.2f}   {:.2f} ".format(new["F2"]*100,  auxillary["F1_class"][-1]*100)

            aveargeString = "{:.2f}   {:.2f}   {:.2f}   {:.2f}   {:.2f}   {:.2f}   {:.2f}   {:.2f}   {:.2f}   {:.2f}".format(main["mF1"]*100, main["MF1"]*100, main["OF1"]*100, main["OP"]*100, main["OR"]*100, main["CF1"]*100, main["CP"]*100, main["CR"]*100, main["EMAcc"]*100, main["mAP"]*100)
            
            classF1String = "   ".join(["{:.2f}".format(x*100) for x in auxillary["F1_class"]])
            classF2String = "   ".join(["{:.2f}".format(x*100) for x in new["F2_class"]])
            classPString = "   ".join(["{:.2f}".format(x*100) for x in auxillary["P_class"]])
            classRString = "   ".join(["{:.2f}".format(x*100) for x in auxillary["R_class"]])
            classAPString = "   ".join(["{:.2f}".format(x*100) for x in auxillary["AP"]])

            with open(os.path.join(outputPath, '{}_latex.txt'.format(outputName)), "w") as text_file:
                text_file.write("New metrics: " + newString + "\n")
                text_file.write("ML main metrics: " + aveargeString + "\n")
                text_file.write("Class F1: " + classF1String + "\n")
                text_file.write("Class F2: " + classF2String + "\n")
                text_file.write("Class Precision: " + classPString + "\n")
                text_file.write("Class Recall: " + classRString + "\n")
                text_file.write("Class AP: " + classAPString + "\n")
    print("calculate results done!")

def operate_csv(args):
    score_path = args["score_path"]
    list = os.listdir(score_path)
    predict_txt_name = ''
    for txt in list:
        if txt.split("_")[0]=='saved':
            predict_txt_name = txt
    gtPath = os.path.join(score_path, predict_txt_name)
    result = os.path.join(score_path, 'result')
    if not os.path.exists(result):
        os.mkdir(result)
    new_file = os.path.join(result, 'predict_'+args["val_csv_name"])
    if not os.path.exists(new_file):
        os.mknod(new_file)
    targetPath = args["gt_path"]
    img_path = os.path.join(targetPath, args['val_csv_name'])  # "valAll_sub_0.0625.csv")
    img = pd.read_csv(img_path, sep=",", encoding="utf-8", usecols=["Filename"])
    img_names = img["Filename"].values
    print(predict_txt_name, " number of images:", len(img_names))

    title = ''
    for index_Labels in range(len(Labels)):
        title = title + Labels[index_Labels]
        if index_Labels != len(Labels) - 1:
            title = title + ','

    with open(gtPath, "r+", encoding='utf-8') as f:
        # read content
        csv_read = csv.reader(f)
        with open(new_file, "r+", encoding='utf-8') as new_f:
            # write title
            new_f.write(title + '\n')
            # write contents
            for row in csv_read:
                row = ','.join(row[0].split(" ")[0:len(Labels)])
                new_f.write(row + '\n')

    # insert names of images
    gt = pd.read_csv(new_file, sep=',', encoding='utf-8', usecols=Labels)
    gt.insert(0, "Filename", img_names)
    gt.to_csv(new_file, index=False)

    #os.remove(gtPath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="Val", choices=["Train", "Val", "Test"])
    parser.add_argument("--score_path", type=str,
                        default = r"/output",
                        help="the directory of the infer results")
    parser.add_argument("--gt_path", type=str,
                        default = r"/dataset",
                        help="the root directory of dataset")
                        #default = r"/media/ubuntu/5385f1e6-0f46-4865-90c2-287b9a0f3c16/sy/complete_Sewer-ML")
    parser.add_argument("--val_csv_name", type=str,
                        default=r"SewerML_Val.csv",# SewerML_Val.csv or SewerML_Val_sub_0.0625.csv
                        help="validation csv file name")

    args = vars(parser.parse_args())
    operate_csv(args)
    calculateResults(args)