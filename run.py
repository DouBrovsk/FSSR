from argparse import ArgumentParser
import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))
from FSSR import meta_train, finetuneMaml, MAMLupscale, model_train, upscale
from evaluation import evaluation

## Parser
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--mode', default='meta_train', choices=['meta_train', 'meta_upscale', 'finetune_maml', 'evaluation', 'upscale_video', 'model_train', 'upscale'])
    parser.add_argument('--device', default='cuda_if_available', choices=['cpu', 'cuda', 'cuda_if_available'], help="Leave default to use the GPU if it is available. CPU can't be used for training without changing the code.")
    parser.add_argument('--input', default='../Memopainter/Memopainter/source/train', help="Path to the directory containing the images to upscale. Only used for upscaling or evaluation.")
    parser.add_argument('--output', default='./out/', help="Destination directory for the benchmark in 'evaluation' mode or upscaled images in 'upscale' mode.")
    parser.add_argument('--verbose', default=True, type=bool, choices=[True, False], help="Wether the script print info in stdout.")
    parser.add_argument('--network_name',  default='EDSR', choices=['EDSR'], help="Indicates which network is being used.")
    parser.add_argument('--logger_name',  default='log', help="Name of the logger.")
    parser.add_argument('--num_shot',  default=10, type=int, help="number of images in the support per task")
    parser.add_argument('--batch_size', default=1, type=int, help="Batch size i.e the number of images for each training iteration as an integer.")

    parser.add_argument('--load_weights', default=None, help="Path to the weights to continue training, perform upscaling on a set of images or evaluate performance.")
    parser.add_argument('--save_weights', default='./weights/memopainter.pt', help="Path to save the weights after training (.pth). Only for 'train' mode.")
    parser.add_argument('--train_folder', default='../Memopainter/Memopainter/source/train', help="Path to the folder containing the images of the training set. Only for 'train' mode.")
    parser.add_argument('--valid_folder', default='../Memopainter/Memopainter/source/test', help="Path to the folder containing the images of the validation set. Only for 'train' mode.")
    parser.add_argument('--epoch_nb', default=10, type=int, help="Number of epochs for training i.e the number of times the whole training set is iterated over as an integer. Only for 'train' mode.")
    parser.add_argument('--learning_rate', default=0.0001, type=float, help="Learning rate for training with Adam optimizer. Only for 'meta_train' & 'meta_upscale' mode.")
    parser.add_argument('--meta_learning_rate', default=0.00001, help="Learning rate of the meta training.")
    parser.add_argument('--loss', default='MSE', choices=['MSE', 'perception', 'ultimate'], help="The loss function to use for training. Percepion loss uses a loss network that can be chosen with --loss_network arg. Only for 'train' mode.")
    parser.add_argument('--loss_network', default='vgg16', choices=['vgg16', 'vgg19', 'resnet18'], help="The loss network used for perceptual loss computing. Only for 'train' mode")
    parser.add_argument('--finetune_depth', default=0, type=int, help="Number of parameter tensors to be finetuned in finetune_maml mode. 0 to modify all layers.")
    opt = parser.parse_args()
    

    if opt.mode == 'meta_train':
        meta_train(logger_name=opt.logger_name,train_path=opt.train_folder, 
                   valid_path=opt.valid_folder, batch_size=opt.batch_size,
                   num_shot=opt.num_shot, epoch_nb=opt.epoch_nb, 
                   learning_rate=opt.learning_rate, meta_learning_rate=opt.meta_learning_rate, 
                   save_path=opt.save_weights, verbose=opt.verbose, weights_load=opt.load_weights, 
                   loss_func=opt.loss, loss_network=opt.loss_network, network=opt.network_name)
        
    elif opt.mode == 'finetune_maml':
        finetuneMaml(train_path=opt.train_folder, valid_path=opt.valid_folder,
                     batch_size=opt.batch_size, epoch_nb=opt.epoch_nb, 
                     learning_rate=opt.learning_rate, meta_learning_rate=opt.meta_learning_rate, 
                     load_weights=opt.load_weights, save_weights=opt.save_weights, 
                     finetune_depth=opt.finetune_depth, network=opt.network_name)
        
    elif opt.mode == 'meta_upscale':
        MAMLupscale(in_path=opt.input, out_path=opt.output, weights_path=opt.load_weights, 
                    learning_rate=opt.learning_rate, batch_size=opt.batch_size, 
                    verbose=opt.verbose, device_name=opt.device, network=opt.network_name)
        
    elif opt.mode == 'evaluation':
        evaluation(in_path=opt.input, out_path=opt.output, verbose=True)
        
    elif opt.mode == 'upscale_video':
        pass
    
    elif opt.mode == 'model_train':
        model_train(logger_name=opt.logger_name,train_path=opt.train_folder, 
                    valid_path=opt.valid_folder, epoch_nb=opt.epoch_nb, 
                    batch_size=opt.batch_size, load_weights=opt.load_weights, 
                    save_weights=opt.save_weights)
        
    elif opt.mode == 'upscale':
        upscale(load_weights=opt.load_weights, input=opt.input, out=opt.output)
        
    else:
        raise Exception("Invalid mode. Run this command if you need help: $ python run.py --help")