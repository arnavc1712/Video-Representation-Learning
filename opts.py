import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
    '--video_encoder_path',
    type=str,
    default='./save/encoder_parameter.pth',
    help='path to a pre-trained encoder model')

    parser.add_argument(
    '--info_json',
    type=str,
    default='data/msrvtt_new_info.json',
    # default = 'data/v2c_info.json',
    help='pre-trained video encoder')

    parser.add_argument(
    '--caption_json',
    type=str,
    default='data/caption.json',
    # default='data/V2C_MSR-VTT_caption.json',
    help='path to the processed video caption json')

    parser.add_argument(
    '--s3d_feat_path',
    type=str,
    default='data/msrvtt_s3d.pkl',
    help='path to the msrvtt S3D features')

    parser.add_argument(
    '--s3d_dict_path',
    type=str,
    default='pretrained/s3d_dict.npy',
    # default='data/V2C_MSR-VTT_caption.json',
    help='path to the word list for S3D'
    )

    parser.add_argument(
    '--bert_feats_path',
    type=str,
    default='data/bert_caps.pkl',
    help='path to the BERT caption features'
    )

    parser.add_argument(
    '--s3d_weights',
    type=str,
    default='pretrained/s3d_howto100m.pth',
    # default='data/V2C_MSR-VTT_caption.json',
    help='path to the pretrained weights for S3D'
    )

    parser.add_argument(
    '--feats_dir',
    nargs='*',
    type=str,
    default=['data/feats/resnet152/'],
    help='path to the directory containing the preprocessed fc feats')

    parser.add_argument('--c3d_feats_dir', type=str, default='data/c3d_feats')

    parser.add_argument(
    '--with_c3d', type=int, default=0, help='whether to use c3d features')

    parser.add_argument(
    '--cached_tokens',
    type=str,
    default='msr-all-idxs',
    help='Cached token file for calculating cider score \
                during self critical training.')

    # Model settings
    parser.add_argument(
    "--max_len",
    type=int,
    default=28,
    help='max length of captions(containing <sos>,<eos>)')

    parser.add_argument(
    '--dim_hidden',
    type=int,
    default=768,
    help='size of the rnn hidden layer')

    parser.add_argument(
    '--dim_lang',
    type=int,
    default=768,
    help='dimension of bert sentence embeddings')

    parser.add_argument(
    '--num_head',
    type=int,
    default=8,
    help='number of attention heads')

    parser.add_argument(
    '--num_layers', type=int, default=1, help='number of layers in the Transformers')

    parser.add_argument(
    '--input_dropout_p',
    type=float,
    default=0.2,
    help='strength of dropout in the Language Model RNN')

    parser.add_argument(
    '--rnn_dropout_p',
    type=float,
    default=0.5,
    help='strength of dropout in the Language Model RNN')

    parser.add_argument(
    '--dim_word',
    type=int,
    default=768,
    help='the encoding size of each token in the vocabulary, and the video.'
    )
    parser.add_argument(
    '--dim_inner',
    type=int,
    default=1024,
    help='Dimension of inner feature in Encoder/Decoder.')

    parser.add_argument(
    '--num_layer',
    type=int,
    default=2,
    help='Numbers of layers in transformers.')

    parser.add_argument(
    '--dim_vid',
    type=int,
    default=1024,
    help='dim of features of video frames')


    parser.add_argument(
    '--num_neg_samples',
    type=int,
    default=20,
    help='number of negative distractors to consider for NCE')

    parser.add_argument(
    '--dim_model',
    type=int,
    default=768,
    help='dim of model')

    # Optimization: General

    parser.add_argument(
    '--epochs', type=int, default=6001, help='number of epochs')

    parser.add_argument(
    '--batch_size', type=int, default=128, help='minibatch size')

    parser.add_argument(
    '--grad_clip',
    type=float,
    default=5,  # 5.,
    help='clip gradients at this value')

    parser.add_argument(
    '--self_crit_after',
    type=int,
    default=-1,
    help='After what epoch do we start finetuning the CNN? \
                (-1 = disable; never finetune, 0 = finetune from start)'
    )

    parser.add_argument(
    '--learning_rate', type=float, default=0.1, help='learning rate')

    parser.add_argument(
    '--learning_rate_decay_every',
    type=int,
    default=30,
    help='every how many iterations thereafter to drop LR?(in epoch)')

    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8)

    parser.add_argument(
    '--optim_alpha', type=float, default=0.9, help='alpha for adam')

    parser.add_argument(
    '--optim_beta', type=float, default=0.999, help='beta used for adam')

    parser.add_argument(
    '--optim_epsilon',
    type=float,
    default=1e-8,
    help='epsilon that goes into denominator for smoothing')

    parser.add_argument(
    '--warm_up_steps',
    type=int,
    default=500,
    help='Warm up steps.')

    parser.add_argument(
    '--weight_decay',
    type=float,
    default=5e-4,
    help='weight_decay. strength of weight regularization')

    parser.add_argument(
    '--save_checkpoint_every',
    type=int,
    default=30,
    help='how often to save a model checkpoint (in epoch)?')

    parser.add_argument(
    '--checkpoint_path',
    type=str,
    default='./save',
    help='directory to store check pointed models')

    parser.add_argument(
    '--load_checkpoint',
    type=str,
    default='./save/bleu_38_epoch_240.pth',
    help='directory to load check pointed models')

    parser.add_argument(
    '--gpu', type=str, default='0', help='gpu device number')

    args = parser.parse_args()

    return args
