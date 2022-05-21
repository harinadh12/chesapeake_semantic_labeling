from fileinput import filename
from chesapeake_loader import *
from networks import *
import argparse
import pickle


def create_parser():

    '''
    Create argument parser

    '''
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='Image Translator', fromfile_prefix_chars='@')


    # Required parameters
    parser.add_argument('--fold', default=None, type=int, required=True,help='Fold number')
    parser.add_argument('--verbose', '-v', action = 'count', required=False, help='verbosity level')
    parser.add_argument('--epochs',  default=10, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--cnn_units', nargs='+', default=None, type=int, help='number of filters in conv layers')
    parser.add_argument('--pool', nargs='+', default=None, type=int, help='pool size for the previous conv layer')
    parser.add_argument('--dataset_path', type=str, default='/home/fagg/datasets/pfam', help='path to dataset')
    parser.add_argument('--patience', type=int, default=10, help='patience')
    parser.add_argument('--min_delta', type=float, default=0.001, help='min delta')
    parser.add_argument('--steps_per_epoch', type=int,default=None,help='Steps per epoch')
    parser.add_argument('--learning_rate', default=0.001, type=float,  help='learning rate')
    
    return parser



if __name__=='__main__':

    ''' Main Function '''

    parser = create_parser()
    args = parser.parse_args()

    # Load the data
    train_data = create_dataset( base_dir='/home/fagg/datasets/radiant_earth/pa', partition='train', fold=args.fold, filt='*[012345678]', 
                   batch_size=32, prefetch=2, num_parallel_calls=2 )
    
    valid_data = create_dataset( base_dir='/home/fagg/datasets/radiant_earth/pa', partition='train', fold=args.fold, filt='*[9]',
                     batch_size=32, prefetch=2, num_parallel_calls=2 )

    test_data = create_dataset( base_dir='/home/fagg/datasets/radiant_earth/pa', partition='valid', fold=args.fold, filt='*',
                        batch_size=32, prefetch=2, num_parallel_calls=2 )

    # Create sequential model
    # model = sequential_image_translator(input_shape=(256,256,26), 
    #                                     n_classes=7,
    #                                     conv_units = args.cnn_units, 
    #                                     pool_size= args.pool, 
    #                                     lrate=args.learning_rate)
    
    #create model using model API
    model = u_net_image_translator(input_shape=(256,256,26),
                    n_classes=7,
                    conv_units = args.cnn_units,
                    pool_size= args.pool,
                    lrate=args.learning_rate)

    history = train_model(model, train_data, valid_data, args)

    save_predictions(test_data, model, history, args)

