import sys
from optparse import OptionParser
from pprint import pprint

# check if arguments are given
if len(sys.argv) < 2:
    print 'Error: no arguments are passed'
    sys.exit()

parser = OptionParser(add_help_option=False)

# add options to parser
parser.add_option("-h", "--help", action="help")
parser.add_option('-f', '--flag', type='int', dest='flag', help='Theano Flags to indicate whether model is deep or not', default=0)
parser.add_option('-m', '--max', '--maxlen', type='int', dest='maxlen', help='Specify the max length', default=1300)
parser.add_option('-e', '--epoch', '--epochs', type='int', dest='num_epoch', help='Specify the number of training iterations performed on the network', default=20)
parser.add_option('-b', '--batch', '--batch_size', type='int', dest='batch_size', help='Specify the batch size for the model', default=64)
parser.add_option('--z1', '--z1_size', type='int', dest='z1', help='Specify the size of the first fully connected layer', default=1024)
parser.add_option('--z2', '--z2_size', type='int', dest='z2', help='Specify the size of the second fully connected layer', default=1024)
parser.add_option('--train', type='string', dest='train', help='Specify the training data path')
parser.add_option('--test', type='string', dest='test', help='Specify the testing data path')


# get arguments and display help if given as arg
(options, args) = parser.parse_args()

# execute very deep cnn
if options.flag == 1:
    execfile("conneau.py")
elif options.flag == 0:
    execfile("zhang.py")
else:
    pprint('Model number not specified')

# example use: python main.py -f 1 -m 1300 -e 10 -b 64 --z1 1024 --z2 1024 --train ag_news_csv/train.csv --test ag_news_csv/test.csv
