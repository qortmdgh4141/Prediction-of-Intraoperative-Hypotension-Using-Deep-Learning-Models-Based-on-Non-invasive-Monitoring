from distutils.util import strtobool

import src.utils.data_loader as dl
import src.utils.preprocess as prep
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_load", type=lambda x: bool(strtobool(x)), default=False)

    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dest_path", type=str, required=True)

    parser.add_argument("--time_seq", type=int, default=3000)
    parser.add_argument("--time_step", type=int, default=1000)
    parser.add_argument("--target_seq", type=int, default=30000)
    parser.add_argument("--interval", type=float, default=0.01)
    parser.add_argument("--maxcase", type=int, default=-1)

    args = parser.parse_args()

    attributes = ["CID",
                  "Time",
                  "SNUADC/ECG_II",
                  "Solar8000/NIBP_SBP",
                  "Solar8000/NIBP_MBP",
                  "Solar8000/NIBP_DBP",
                  "SNUADC/PLETH",
                  "Primus/MAC",
                  "Primus/CO2",
                  "BIS/BIS"]

    if args.data_load:
        # 1. load from vital DB framework
        dl.data_load(args.dset_path, attributes, args.maxcase, args.interval)
        prep.make_dataset(
            args.dataset_path,
            time_seq=args.time_seq,
            time_step=args.time_step,
            target_seq=args.target_seq,
            dst_path=args.dest_path
        )
    else:
        prep.make_dataset(
            args.dataset_path,
            time_seq=args.time_seq,
            time_step=args.time_step,
            target_seq=args.target_seq,
            dst_path=args.dest_path
        )

    print("[Done] Data maker.")
