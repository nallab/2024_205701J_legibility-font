import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'forward', 'split_datasets'], help='mode')

    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--n_epochs", type=int, default=30, help="number of epochs of training")
    parser.add_argument("--max_fonts_num", type=int, default=10000)
    parser.add_argument("--checkpoint_root", type=str, default="checkpoint")
    parser.add_argument("--dataset_root", type=str, default="data")
    parser.add_argument("--image_dir_name", type=str, default="images")
    parser.add_argument("--train_font_names_file_name", type=str, default="train_font_names.txt")
    parser.add_argument("--test_font_names_file_name", type=str, default="test_font_names.txt")
    parser.add_argument("--val_font_names_file_name", type=str, default="val_font_names.txt")
    parser.add_argument("--forward_font_names_file_name", type=str, default="fontname.txt")
    parser.add_argument("--ignore_fonts_file_name", type=str, default="eliminate_fonts.txt")
    parser.add_argument("--is_attr2font_dataset", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_loss_csv_path", type=str, default="loss_each_fonts.csv")
    parser.add_argument("--output_loss_csv_dir", type=str, default="output")
    parser.add_argument("--load_weight_path", type=str, default="default")



    return parser
