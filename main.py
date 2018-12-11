import argparse
from text_to_image_GAN_main import Text_to_image_GAN


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # argument of path
    parser.add_argument("--write_gen_model_path", default="/result/t2iGAN_gen")
    parser.add_argument("--write_dis_model_path", default="/result/t2iGAN_dis")
    parser.add_argument("--write_generated_img_path", default="/project/samples")
    parser.add_argument("--write_summary_path", default="/project/summaries")
    parser.add_argument("--train_img_data_path", default="/data/images")
    parser.add_argument("--train_txt_data_path", default="/data/text_c10")
    parser.add_argument("--train_meta_path", default="/data/trainvalclasses.txt")
    parser.add_argument("--test_meta_path", default="/data/testclasses.txt")
    parser.add_argument("--text_encoder_ckpt", default="/model/DS_SJE")

    # argument of text encoder
    parser.add_argument("--text_represent_dim", default=512, help="put text representation dimension", type=int)
    parser.add_argument("--num_caption", default=5, type=int)
    parser.add_argument("--alpha_size", default=70, type=int)

    # argument of hyperparameter
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--z_input_size", default=100, type=int)
    parser.add_argument("--txt_embed_size", default=128, type=int)
    parser.add_argument("--x_input_size", default=64, type=int)
    parser.add_argument("--num_init_filter", default=1024, type=int)
    parser.add_argument("--num_epoch", default=1000, type=int)
    parser.add_argument("--D_learning_rate", default=0.0002, type=float)
    parser.add_argument("--G_learning_rate", default=0.0002, type=float)
    parser.add_argument("--learning_rate_decay", default=0.5, type=float)
    parser.add_argument("--learning_rate_decay_every", default=100, type=float)
    parser.add_argument("--lambda_h", default=10, type=int)
    parser.add_argument("--num_channel", default=3, type=int)
    parser.add_argument("--cnn_represent_dim", default=1024, type=int)

    # argument of data loader
    parser.add_argument("--train_img_data_type", default="*.jpg")
    parser.add_argument("--train_txt_data_type", default="*.npy")
    parser.add_argument("--buffer_size", default=5000, type=int)
    parser.add_argument("--prefetch_multiply", default=10, type=int)
    parser.add_argument("--multi_process_num_thread", default=12, type=int)

    # turn on or off CNN correctioner
    parser.add_argument("--CNN_correctioner", default="none")

    args = parser.parse_args()

    model = Text_to_image_GAN(args=args)
    model.train()
