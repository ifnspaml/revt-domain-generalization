import re
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='reads mIoU values from test result')

    parser.add_argument('--filename', type=str, default="test_xxx_on_yyy.log", help='name of the output config file')
    parser.add_argument('--num-models', default=3, type=int, help='number of samples per dataset. Default is 3 models per dataset.')
    parser.add_argument('--dataset-order', type=str, nargs='+', default=["Cityscapes", "bdd", "mapillary", "acdc", "kitti", "gta_val"]
        , help='A list the which defines which datasets are collected and in which order they are listed')

    args = parser.parse_args()
    return args

def get_miou_from_file(filename):
    with open(filename) as f:
        file = f.readlines()
    
    if not isinstance(file, str):
        res = ""
        for line in file:
            res += line
        file = res
    
    regex = "\\|[0-9\. ]*\\|[0-9\. ]*\\|[0-9\. ]*\\|"
    findings = re.findall(regex, file)
    
    findings = [float(f.replace(" ", "")[1:-1].split("|")[1]) for f in findings]
    return findings[0]

def main():
    
    args = parse_args()
    datasets = args.dataset_order #["Cityscapes", "bdd", "mapillary", "acdc", "kitti", "gta_val"]
    results = []
    all = []
    for dataset in datasets:
        results.append([])
        for i in range(args.num_models):
            file = args.filename
            file = file.replace("test_xxx", f"test_{chr(ord('a')+i)}")
            file = file.replace("on_yyy", f"on_{dataset}")
            mIoU = get_miou_from_file(file)
            results[-1].append(mIoU)
            all.append(mIoU)
        
    results.append(all)
    datasets.append("all")
    
    latex_string = ""
    for i, res_set in enumerate(results):
        print(f"Dataset: {datasets[i]}")
        print(f"    data row: {res_set}")
        print(f"    mean: {round(np.mean(res_set),2)}, std: {round(np.std(res_set), 2)}")
        latex_string += str(f"${round(np.mean(res_set),1)} \\pm {round(np.std(res_set), 1)}$ & ")
    print(f"Latex String:\n{latex_string[0:-2]}\\\\")
if __name__ == '__main__':
    main()