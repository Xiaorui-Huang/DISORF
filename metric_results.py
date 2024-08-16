import argparse
import os

import pandas as pd
from tbparse import SummaryReader


def dump_tb(tb_folder):
    # if metrics.csv already exists, load it in and print it 
    if os.path.exists(f"{tb_folder}/metrics.csv"):
        df = pd.read_csv(f"{tb_folder}/metrics.csv", index_col=False)
        columns_to_drop = [col for col in df.columns if 'Unnamed' in col or not col]

        # Drop those columns
        df = df.drop(columns=columns_to_drop)
        print(df)
        return

    # tb_file starts with "events.out.tfevents."
    df = SummaryReader(tb_folder, pivot=True).scalars # pandas dataframe

    metrics = [
    'Eval Images Metrics Dict (all images)/lpips',
    'Eval Images Metrics Dict (all images)/psnr',
    'Eval Images Metrics Dict (all images)/ssim',
    'Eval Images Metrics Dict (all images)/lpips_std',
    'Eval Images Metrics Dict (all images)/psnr_std',
    'Eval Images Metrics Dict (all images)/ssim_std',
    ]

    valid_rows = df[metrics[0]].notnull()

    extracted_df = df[metrics][valid_rows]

    # remove prefix
    extracted_df.columns = extracted_df.columns.str.replace("Eval Images Metrics Dict (all images)/", "")

    # save to csv
    extracted_df.to_csv(f"{tb_folder}/metrics.csv", index=False)
    print(extracted_df)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--keyword", type=str, default="")
    parser.add_argument("-u","--union_keyword", action="store_true")
    parser.add_argument("--exclude", type=str, default="")
    parser.add_argument("-c", "--clean", action="store_true")
    args = parser.parse_args()
    
    keywords = args.keyword.split(",")
    exclude_keywords = list(filter(lambda x: x != "", args.exclude.split(",")))
    
    print("Eval for the following directories:")
    for dir_name, subdir_list, file_list in os.walk(args.folder):
        for file in file_list:
            if file.startswith("events.out.tfevents."):
                keywords_matched = all([kw in dir_name for kw in keywords]) if args.union_keyword else any([kw in dir_name for kw in keywords])
                if keywords_matched and not any([exclude_kw in dir_name for exclude_kw in exclude_keywords]):
                    if args.clean and "nerfstudio_models" not in subdir_list:
                        print(f"No nerfstudio_models in the folder, skip. {dir_name}")
                        os.system(f"rm -rI {dir_name}")
                        continue
                    print("\t", dir_name)

    # enumerate all subfolders, if any of them contains the file "events.out.tfevents.*", then dump it
    for dir_name, subdir_list, file_list in os.walk(args.folder):
        for file in file_list:
            if file.startswith("events.out.tfevents."):
                keywords_matched = all([kw in dir_name for kw in keywords]) if args.union_keyword else any([kw in dir_name for kw in keywords])
                if keywords_matched and not any([exclude_kw in dir_name for exclude_kw in exclude_keywords]):
                    print(dir_name)
                    try:
                        dump_tb(dir_name)
                    except Exception as e:
                        print(f"Error {e} in dumping tb, skip {dir_name}.")

if __name__ == "__main__":
    main()