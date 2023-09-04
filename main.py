import warnings

import torch

torch.set_printoptions(profile='full')

import re
import datetime

import arguments
from utils import *

# parameters
args_input = arguments.get_args()
NUM_QUERY = args_input.batch
NUM_INIT_LB = args_input.initseed
NUM_ROUND = int(args_input.quota / args_input.batch)
DATA_NAME = args_input.dataset_name
STRATEGY_NAME = args_input.ALstrategy
SAVE = args_input.save_model
NET = args_input.net

SEED = args_input.seed
os.environ['TORCH_HOME'] = './basicmodel'
os.environ["CUDA_VISIBLE_DEVICES"] = str(args_input.gpu)

# fix random seed
np.random.seed(SEED)
torch.manual_seed(SEED)

# device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# if use_cuda:
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     torch.cuda.manual_seed_all(SEED)

if use_cuda:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

def main():
    # file check
    if not os.path.exists('ckpt'):
        os.mkdir('ckpt')
    # recording
    sys.stdout = Logger(
        os.path.abspath('') + '/logfile/' + DATA_NAME + '_' + STRATEGY_NAME + '_' + str(NUM_QUERY) + '_' + str(
            NUM_INIT_LB) + '_' + str(args_input.quota) + '_normal_log.txt')
    warnings.filterwarnings('ignore')

    # start experiment

    iteration = args_input.iteration

    all_acc = []
    all_res = []
    acq_time = []

    # repeate # iteration trials
    while (iteration > 0):
        iteration = iteration - 1

        # data, network, strategy
        args_task = args_pool[DATA_NAME]

        if NET == 'sganet':
            dataset = get_dataset_sganet(args_input.dataset_name, args_task)  # load dataset
        else:
            dataset = get_dataset(args_input.dataset_name, args_task)  # load dataset

        if args_input.ALstrategy == 'LossPredictionLoss' or args_input.ALstrategy == 'TA-VAAL':
            net = get_net_lpl(args_input.dataset_name, args_task, device, args_input.ALstrategy)  # load network
        elif args_input.ALstrategy == 'WAAL':
            net = get_net_waal(args_input.dataset_name, args_task, device)  # load network
        else:
            net = get_net(args_input.dataset_name, args_task, device, NET)  # load network
        strategy = get_strategy(args_input.ALstrategy, dataset, net, args_input, args_task)  # load strategys

        start = datetime.datetime.now()

        # generate initial labeled pool
        sampler = get_sampler(args_input.initsampler, dataset, args_input, args_task)
        sampler.sample(args_input.initseed)
        print('initial seed {}'.format(args_input.initseed))

        # record acc performance
        acc = np.zeros(NUM_ROUND + 1)
        round_res = []

        # only for special cases that need additional data
        new_X = torch.empty(0)
        new_Y = torch.empty(0)

        # print info
        print(DATA_NAME)
        print('RANDOM SEED {}'.format(SEED))
        print(type(strategy).__name__)

        # round 0 accuracy, Crowd count net will return res, image classification will return None
        if args_input.ALstrategy == 'WAAL':
            res = strategy.train(model_name=args_input.ALstrategy)
        # elif args_input.ALstrategy == 'PSSW':
        #     res = strategy.train(model_name=args_input.ALstrategy)
        # elif args_input.ALstrategy == 'CrowdSemi' or args_input.ALstrategy == 'CrowdIRAST':
        #     res = strategy.train(model_name=args_input.ALstrategy)
        # elif args_input.ALstrategy == 'CrowdProject' or args_input.ALstrategy == 'HeadCount':
        #     res = strategy.train(model_name=args_input.ALstrategy)
        else:
            res = strategy.train()
        # round rd accuracy
        if res is not None:
            acc[0] = res['mae']
            print('testing accuracy {}'.format(acc[0]))
        else:
            preds = strategy.predict(dataset.get_test_data())
            acc[0] = dataset.cal_test_acc(preds)['acc']
            # For classification, it is acc
            print('testing accuracy {}'.format(acc[0]))
            print('\n')
        round_res.append(res)
        # round 1 to rd
        for rd in range(1, NUM_ROUND + 1):
            print('Round {}'.format(rd))
            high_confident_idx = []
            high_confident_pseudo_label = []
            # query
            if 'CEALSampling' in args_input.ALstrategy:
                q_idxs, new_data = strategy.query(NUM_QUERY, rd, option=args_input.ALstrategy[13:])
            else:
                 q_idxs = strategy.query(NUM_QUERY)

            # update
            strategy.update(q_idxs)

            # train
            if 'CEALSampling' in args_input.ALstrategy:
                res = strategy.train(new_data)
            elif args_input.ALstrategy == 'WAAL':
                res = strategy.train(model_name=args_input.ALstrategy)
            # elif args_input.ALstrategy == 'PSSW':
            #     res = strategy.train(model_name=args_input.ALstrategy)
            # elif args_input.ALstrategy == 'CrowdProject' or args_input.ALstrategy == 'HeadCount':
            #     res = strategy.train(model_name=args_input.ALstrategy)
            else:
                res = strategy.train()

            # round rd accuracy
            if res is not None:
                acc[rd] = res['mae']
                print('testing accuracy {}'.format(acc[rd]))
            else:
                preds = strategy.predict(dataset.get_test_data())
                acc[rd] = dataset.cal_test_acc(preds)['acc']
                # For classification, it is acc
                print('testing accuracy {}'.format(acc[rd]))
                print('\n')

            round_res.append(res)

        # torch.cuda.empty_cache()

        # print results
        print('SEED {}'.format(SEED))
        print(type(strategy).__name__)
        print(acc)
        all_acc.append(acc)
        all_res.append(round_res)
        print(f'Trials remaining {iteration}')

        if SAVE:
            # save model
            timestamp = re.sub('\.[0-9]*', '_', str(datetime.datetime.now())).replace(" ", "_").replace("-",
                                                                                                        "").replace(
                ":",
                "")
            # if modelpara doesnot exist, create
            if not os.path.exists('./modelpara'):
                os.makedirs('./modelpara')

            model_path = './modelpara/' + timestamp + DATA_NAME + '_' + STRATEGY_NAME + '_' + str(
                NUM_QUERY) + '_' + str(
                NUM_INIT_LB) + '_' + str(args_input.quota) + '.params'
            torch.save(strategy.get_model().state_dict(), model_path)

        end = datetime.datetime.now()
        acq_time.append(round(float((end - start).seconds), 3))

    # cal mean & standard deviation
    if not os.path.exists('./results'):
        os.makedirs('./results')
    acc_m = []

    # save readable result
    if not os.path.exists(f'./results/{DATA_NAME}'):
        os.makedirs(f'./results/{DATA_NAME}')

    result_path = f'/results/{DATA_NAME}'

    custom_tag = '_' + args_input.tag if args_input.tag else ''
    file_name_res = DATA_NAME + '_' + STRATEGY_NAME + '_' + str(NUM_QUERY) + '_' + str(NUM_INIT_LB) + '_' + str(
        args_input.quota) + '_normal_res' + custom_tag + '.txt'
    file_res = open(os.path.join(os.path.abspath('') + result_path, '%s' % file_name_res), 'w')

    file_res.writelines('dataset: {}'.format(DATA_NAME) + '\n')
    file_res.writelines('AL strategy: {}'.format(STRATEGY_NAME) + '\n')
    file_res.writelines('number of labeled pool: {}'.format(NUM_INIT_LB) + '\n')
    file_res.writelines('number of unlabeled pool: {}'.format(dataset.n_pool - NUM_INIT_LB) + '\n')
    file_res.writelines('number of testing pool: {}'.format(dataset.n_test) + '\n')
    file_res.writelines('batch size: {}'.format(NUM_QUERY) + '\n')
    file_res.writelines('quota: {}'.format(NUM_ROUND * NUM_QUERY) + '\n')
    file_res.writelines('time of repeat experiments: {}'.format(args_input.iteration) + '\n')
    avg_acc = np.mean(np.array(all_acc), axis=0)
    std_acc = np.std(np.array(all_acc), axis=0)

    for i in range(len(avg_acc)):
        tmp = 'Size of training set is ' + str(NUM_INIT_LB + i * NUM_QUERY) + ', ' + 'accuracy is ' + str(
            round(avg_acc[i], 4)) + '.' + '\n'
        file_res.writelines(tmp)

    # result
    for i in range(len(all_acc)):
        acc_m.append(get_aubc(args_input.quota, NUM_QUERY, all_acc[i]))
        print(str(i) + ': ' + str(acc_m[i]))
        file_res.writelines(str(i) + ': ' + str(acc_m[i]) + '\n')
    mean_acc, stddev_acc = get_mean_stddev(acc_m)
    mean_time, stddev_time = get_mean_stddev(acq_time)

    print('mean AUBC(acc): ' + str(mean_acc) + '. std dev AUBC(acc): ' + str(stddev_acc))
    print('mean time: ' + str(mean_time) + '. std dev time: ' + str(stddev_time))
    file_res.writelines('mean acc: ' + str(mean_acc) + '. std dev acc: ' + str(stddev_acc) + '\n')
    file_res.writelines('mean time: ' + str(mean_time) + '. std dev acc: ' + str(stddev_time) + '\n')
    file_res.writelines('-----------------------------------------------------' + '\n')
    file_res.writelines('Individual round results:' + '\n')
    # Individual results
    for i in range(len(all_acc)):
        for j in range(len(all_acc[i])):
            temp = 'Round ' + str(j) + ': ' + str(all_acc[i][j]) + '\n'
            file_res.writelines(temp)
        file_res.writelines('-----------------------------------------------------' + '\n')

    file_res.writelines('All Metrics' + '\n')
    for i in range(len(all_res)):
        file_res.writelines(f"--------------Iteration {i}--------------" + '\n')
        for idx, res in enumerate(all_res[i]):
            file_res.writelines(f"---Round {idx}---" + '\n')
            for key, value in res.items():
                file_res.writelines(f"{key}: {value}" + '\n')
        file_res.writelines('-----------------------------------------------------' + '\n')

    file_res.writelines('seed: ' + str(SEED) + '\n')
    file_res.writelines('-----------------------------------------------------' + '\n')
    file_res.writelines('Date time: ' + str(datetime.datetime.now()) + '\n')
    file_res.writelines('-----------------------------------------------------' + '\n')

    # Save not readable result (For plot figure)
    file_unreadable_name = 'P_' + DATA_NAME + '_' + STRATEGY_NAME + '_' + str(NUM_QUERY) + '_' + str(NUM_INIT_LB) + '_' + str(
        args_input.quota) + custom_tag + '.txt'
    file_unreadable = open(os.path.join(os.path.abspath('') + result_path, '%s' % file_unreadable_name), 'w')
    file_unreadable.writelines(DATA_NAME + '\n')
    file_unreadable.writelines(STRATEGY_NAME + '\n')

    # Write acc and std
    for i in range(len(avg_acc)):
        file_unreadable.writelines(str(NUM_INIT_LB + i * NUM_QUERY) + ',' + str(round(avg_acc[i], 4))+ ',' + str(round(std_acc[i], 4)) + '\n')

    file_res.close()
    file_unreadable.close()
    print('File saved as ' + file_name_res)
    print('data time' + str(datetime.datetime.now()))


if __name__ == '__main__':
    main()
