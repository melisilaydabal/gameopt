import argparse

def parse_args():
    '''
    Argument parser
    '''

    parser = argparse.ArgumentParser()
    # --Game
    parser.add_argument("--nb_experiments", type=int, default=18,
                        help="The number of experiments (active learning + games) to perform, default is 1.")
    parser.add_argument("--nb_games", type=int, default=3,
                        help="The number of games to play in each iteration, default is 1.")
    parser.add_argument("--nb_players", type=int, default=4,
                        help="The number of players in the game, default is 4.")
    parser.add_argument("--eq_type", type=str, default='CCE', choices=['CCE', 'BR'],
                        help="Equilibrium type to be computed, "
                             "possible algorithms: {CCE, BR},"
                             "default is 'CCE'.")
    parser.add_argument("--is_surr_model", action='store_true',
                        help="Whether to use surrogate utility function, if not mentioned it is taken as False.")
    parser.add_argument("--is_adaptive_grouping", action='store_true',
                        help="Whether to consider adaptive grouping, if not mentioned it is taken as False.")
    parser.add_argument("--is_decay_lr", action='store_true',
                        help="Whether to decay the learning rate, if not mentioned it is taken as False.")
    parser.add_argument("--is_true_baseline", action='store_true',
                        help="Whether to use sampled ground truth as the dataset, if not mentioned it is taken as False.")
    parser.add_argument("--nb_groups", type=int, default=1,
                        help="The number of players in each player group, default is 1.")
    parser.add_argument("--model_name", type=str, default='gp_',
                        help="The name of the trained utility model to use in the game, default is 'gp'.")

    # --Algorithm-related
    parser.add_argument("--nb_rounds", type=int, default=400,
                        help="The number of rounds to play the game, default is 10.")
    parser.add_argument("--algorithm", type=str, default='Hedge', choices=['_', 'MW', 'Hedge', 'Hedge_Mixed', 'FW',
                                                                           'GP-UCB', 'Random', 'Iter_Random',
                                                                           'IBR-UCB', 'IBR-Fitness', 'IBR-UCB_cycle',
                                                                           'PR', 'Discrete_Local', 'Discrete_Local_Best',
                                                                           'LAMBO', 'LADDER', 'LATENTOPT'],
                        help="Algorithm to compute chosen EQ, "
                             "possible algorithms: {MW, Hedge, Hedge_Mixed, FW (for Frank-Wolfe)},"
                             " default is 'Hedge'.")
    parser.add_argument("--lr", type=float, default=0.5,
                        help="The learning rate (multip_factor) for the Multiplicative-Weights, Hedge and Frank-Wolfe algorithms. "
                              "It should be in range [0,1]; for MW, It should be in range (0,0.5], default is 0.5.")
    parser.add_argument("--lr_decay_rate", type=float, default=0.1,
                        help="The learning rate decay factor, default is 0.1.")
    parser.add_argument("--pr_lr", type=float, default=0.1,
                        help="The learning rate for the PR parameter training. ")
    # parser.add_argument("--mc_batch_size", type=int, default=1024,
    #                     help="The batch size to estimate PO and its gradient via MC, default is 1024.")
    parser.add_argument("--mc_batch_size", type=int, default=1024,
                        help="The batch size to estimate PO and its gradient via MC, default is 1024.")
    parser.add_argument("--nb_mc_iterations", type=int, default=10,
                        help="The number of iterations to train parameters of probabilistic objective (PO) for PR baseline. "
                             "default is 10.")


    # --GP Regression-related
    parser.add_argument("--training_iter", type=int, default=200,
                        help="The number of pre-training iterations for the surrogate model. "
                              "default is 200.")
    parser.add_argument("--test_size", type=float, default=0.99,
                        help="The fraction to determine the size of the test data."
                              "default is 0.99.")
    kernel_map = {'rbf': gpytorch.kernels.RBFKernel(),
                  'matern': gpytorch.kernels.MaternKernel(),
                  'linear': gpytorch.kernels.LinearKernel(),
                  'shifted_rbf': ShiftedRBFKernel(),
                  'subsequence_string': SubsequenceStringKernel(),
                  'ss': None,
                  'rbfss': None,
                  'combined': None}   #to avoid empty initialization
    parser.add_argument("--kernel", choices=kernel_map.keys())
    parser.add_argument("--beta", type=float, default=2.0,
                        help="The parameter of the UCB acquisition function"
                             "default is 2.0.")
    parser.add_argument("--acquisition", type=str, default='ucb', choices=['', 'ucb'],
                        help="The type of acquisition function"
                             "default is ''.")
    parser.add_argument("--nb_iterations", type=int, default=50,
                        help="The number of online/active learning iterations. "
                              "default is 50.")
    parser.add_argument("--nb_sample", type=int, default=5,
                        help="The number of points to sample in each active learning iteration. "
                              "default is 5.")
    parser.add_argument("--last_nb_rounds", type=int, default=11,
                        help="The last number of rounds to sample potential equilibria. "
                              "default is 11.")
    parser.add_argument("--initial_strategy", type=str, default='',
                        help="Initial strategy of the game for method comparison.")
    # 0 log fitness initial strategies
    # parser.add_argument("--list_initial_strategy", nargs="+", default=['VDPM', 'AMIK', 'TDCK', 'DDYT', 'GDIV', 'TLQV',
    #                                                                    'VDSW', 'PPNV', 'ASKV', 'ASEV', 'DDLH', 'DDSW',
    #                                                                    'VAKV', 'KYTV', 'PDWL', 'VPDV', 'GMEV', 'EGGD'],
    #                     help="List of initial strategies of the game for method comparison for each experiment (split).")
    # best-so-far initial strategies
    parser.add_argument("--list_initial_strategy", nargs="+", default=['YHAA', 'MQLG', 'IQGA', 'LKCG', 'FWAS', 'LWLT',
                                                                       'ECLG', 'YKCM', 'SHCA', 'ALAA', 'RICA', 'WALA',
                                                                       'LYGI', 'DCGL', 'WFMT', 'WFGL', 'PADD', 'WGFA'],
                        help="List of initial strategies of the game for method comparison for each experiment (split).")
    # For GB1(4): 500 initial training set
    # parser.add_argument("--list_initial_strategy", nargs="+", default=['YHAA', 'YYAG', 'QKCA', 'WHCG', 'FIMA', 'FWCA',
    #                                                                    'VIAA', 'VYGN', 'SHCA', 'YRFG', 'YRLC', 'WALA',
    #                                                                    'VAAA', 'YICA', 'EWGM', 'VYGN', 'EICA', 'WCFG'],
    #                     help="List of initial strategies of the game for method comparison for each experiment (split).")
    # # For GB1(4): 1000 initial training set
    # parser.add_argument("--list_initial_strategy", nargs="+", default=['YHAA', 'YYAG', 'FWGS', 'WQCA', 'IYCA', 'FWCA',
    #                                                                    'WWLG', 'WWCG', 'IALG', 'YRFG', 'IYCG', 'WALA',
    #                                                                    'VAAA', 'YICA', 'WHCA', 'ERLG', 'EICA', 'WCFG'],
    #                     help="List of initial strategies of the game for method comparison for each experiment (split).")

    parser.add_argument("--list_initial_strategy_gb1_55", nargs="+", default=['FDMSCVDAFWCTAHLVHLECFKNQNVCWFEDKHIACLVNTIFEDWMKCWISIHNQ',
                                                                              'EVIMKWIPMVRQAWADMACIQAMYQIYFMPATRRSQCSILYGHDYFWCIQETYMP',
                                                                              'SPCIGHKYHQFTKRWIIWIDEAMKFCYGYQIRRIYWFCHGTNTNGMCRHWKEQVS',
                                                                              'RFTIWDQEGFEIELECQWMTAHNVMLLWDICHRAQGFMEVGWYCMHMCKGNRKAY',
                                                                              'VDQYFPLAMYTKWIFVDGWGVFTEPQLETCPQPEGICQTDEGKVFLLQFFAAMNL',
                                                                              'MWLFTDGYIAWCCAMHGICTFYLQTDWMYLWQQSCVNIYGMITSACGMMMFYIHD',
                                                                              'KILCTHAVLALYWEYVKMFWPSLHFKCGVEDLWYAQVHAFQFWHGYYDTFGNMCN',
                                                                              'LGQHGELGCSIISNRCSCDVKCWMYLVFTPPQIFRWTNWNMSVKVNGGDCMFYGD',
                                                                              'FAQAHQVDQQEGLNKHTGMQILEKMTYAMDTGRRMVLSNFWTCAKAAGSMNDFEI',
                                                                              'ETMKTSCAFTELNYKCGQWCAMYNNTSENCLNVCNWFLIRVNNPDREHYYQFPGI'],
                        help="List of initial strategies of the game for method comparison for each experiment (split).")

    # best-so-far initial strategies
    parser.add_argument("--list_initial_strategy_halogenase", nargs="+", default=['MIS', 'DIG', 'SPC', 'MGT', 'AGL', 'AGL',
                                                                       'WCP', 'DPL', 'FPS', 'AGM', 'MAP', 'MQA',
                                                                       'MNA', 'MQA', 'FPS', 'MGV', 'MAN', 'MNA'],
                        help="List of initial strategies of the game for method comparison for each experiment (split).")
    parser.add_argument("--list_initial_strategy_gfp", nargs="+",
                        default=['QCFFDVFQGKSRPWCCTKEIRLMAAILNFNTEWLNPVFIGRWMAMMVDTLHMMLSTEWMAKDRVRKTWRCIWQHLIHINPFSVQHVTWWHKAMQPQPEETMVTNPYYLYVGFAEYALVPKRGWMPNLDMECTNHYWQKHFAMCIKKALTRDWKVCIWVTTDTWLIFPMCRVPEYFDQGIHISQRVHLMHSTQGQMNAETQHEEKQQSVRNKDPRMSQCRNFHRLLDARQCIGEFMKNW',
                                 'IFSNWTISMNTDIKGVFEDKWYPTWLSGVVMHLTGAKVQYDDWIMVNFWVYSVKNGVWRTLKVINPWVAILIYDNCLCRCVGPRMNEFPCKSQFIMWLYHGIMTTWVKDNIGAYLCVWVWMRPCTQCQKLNGCYHLRMPCQYVAWGLSQHLDWHWANHYFYEFTLIMGRAWRLCDIPWDDREFQDKRDQKRIDHDYYLFCPGEMHMLRRDGAMHCRKSVWWSLIYQDRIIGNFSDETD',
                                 'CRNEFILFTQHWHSRAIWEYKCMPFQHEGGVVDMSLKEFAGCINMNIFSKHPFWPMTIPMVAWGPEVLHDRSSSDHPIGHKLFNNKIFVKITKHFLTAEHMKHQRHMAETTHTTQCVQPMRVDMYPCVYGIYVIETWGVSMFEDDDLMEMEDTMESEMDAPYMFKPFLVKARASEIYCLSTFSMIHYVWRDPRNALMGDVPLKNEIWGEHARYPGCRRPYFRNDIAHAAKSGACKCAT',
                                 'SNVNRVCEVEGAFVEEFKQFLPIYLRGQQIIGTPNGNNLTIWHAAWWWLAADHVCIKVKNTEYFSLFLWHMQCSPLKLMHNSFPMPPIDTNLETHMKDTQIVWEYFAGLEQWEGSAKPRPWPTPRHERGYWENIVIQNFMWPCFEPGHEQFKYVRGHGDAPAHMTPFCKRIVYDEMGMRSLYSCCHKFDFDCDIDLAHRGAEMYTFNLVMTCAFDFSEATNTQHADLKKLHMTWVCDR',
                                 'KGRVCNSHARSMKYQHVGPMQCMCHPDTSTNVMGFIGLWVMYIRIGWNESTFRDCAYSVRSRNIPHFHEIMRVIGFYTMDVGPTPDYMSCRLDLIYHGDEICLNWNYIFWEIRSQCVESLDRVFSVKFCQPGITMQCEKKAVQFKHDVHVFDFYDPLGMPADVHEEWVLHSFSFRRDCKMMFELVNDWLKDCISFVTFCKWYDHWHGPVDCWHKRRMVASIIQMAFEMRFPLVEEREH',
                                 'DADMWFGYAPYWWVIFPCWCKWDPKFQLRLSNAAIMGICCKKITFGTSNVVIMIGHICAMCQQNCEWVKLKKFCIDVWPQQCIPHNPWKCVVRQCRHCMEWNGTGWPPFYSASQARWSAHPKQQGDVGQSPQQHEYQTKMVVIFMKFIWHLIDNPQHNHDKIDWTTNYFEITNGWNAKQSEWVWSPMVKMRCCVWWMGQNIRQTIRRYLPYEGTMHKKVFTAVCSFEHNPNILHVMAT',
                                 'WGTKLWMETFYLPYMGPHHVQWRIYMTDGKRDKNAFFTLDHISVQWYRGVTHTAIVHSSCCQRWFITHCNAFFDTGFHSQAQMPWTFETLRFFFIYCWQIHWQVACCLRYHRNMVDSIMEANARTLSTYSHCPHNICEMHRYIYCFVKYSMDRVADPMDEEYNCVHLCDMCRYIWDHYQFTTNAVSLIHPSFFLVLQVKQPECVDTPYYDAFQPHDSKTYNWNPQEHEEESYFNLEMK',
                                 'MKHLSKAVHHVYWRPRCCQKEGFKELSAWSQVDFHFSEINQKAIHSYDNMFGMCQNEMTDEVSIMGNNKHKDIMCWKHPIVRHELDWLTAYLHMAHYGLGDCLARGAMALKNPGGARFDHAHPRVVKLFMSIAVTSWHQHLNFCRDTHRQLLLWGLEIYVVFPKAFLNVRPIDSQHISTLLDIIFSYGVMDHQPTKGGTNRYWAIFIHQPMCIRFSLVKGLKIFPSQEDHDAWQLIGD',
                                 'CHSETMWLDYYMKCVASRMWGTRYCIVFSHLEMPGNNTEHFFTSRGESIKARTTGVADFVWNLFQLVHFIAMIIMYDPVCRHLFASWNLEKMQMCGHLANRKISMLMMVHTGTQICWLTWAWQKTECMACYTAQADLMPDADYPGHGRWFAKWCNDLLLFHNNQWWVAPRTALACFKANLQRDIDAQTTVMCDRTYINKYAKILIEIDQILRFVQTSDRQDCWIDGNSRWFYCGRPFA',
                                 'YTIPVWTEWPDREVWGEQHLTHLSYCWTETFCRMYIQFSWEFQMYMIQVTTMEHNICLKTIYIWQINCCAYLAPGHNAHNLNWDIWHGCHAATTEEQIMMANLTKGSKWHIHRHSYYKCCWHPEPFNFYTFVLPAFHWIWVIEHFPIMIKPVMSVYTQIWEWCRHFRALEDFCGLALQNPKIYYQPEKNARTKAANWNTMSEQICHTKNIFCRPVRRGTKHINPRMIERAVYKNRICK'],
                        help="List of initial strategies of the game for method comparison for each experiment (split).")

    parser.add_argument("--list_player_indexes", nargs="+",
                        # default=[23,40,44,49,45,34,20,47,46,38],
                        # default=['02', '13'],
                        default=[10,18,22,37,67,78,196,112],
                        # default=[10,18,22,37,67,78],
                        # default=[0, 1],
                        # default=[0],
                        help="List of player indexes (protein sites) to play the game with nb_players < length of sequence"
                             "default is: [23,40,44,49,45,34,20,47,46,38] for GB1(55)_10 and "
                             "['12', '34'] for GB1(4) w adaptive grouping"
                             "")

    # Directories
    parser.add_argument("--dir_dataset", type=str, default='../datasets/',
                        help="Directory that store datasets, default is '../datasets/'.")
    parser.add_argument("--dir_plots", type=str, default='../workbench/plots/',
                        help="Directory that store plots, default is '../workbench/plots/'.")
    parser.add_argument("--dir_workbench", type=str, default='../workbench/',
                        help="Directory that store workbench, default is '../workbench/'.")
    parser.add_argument("--dir_dump", type=str, default='../workbench/',
                        help="Directory that store dump (results), default is '../workbench/'.")
    parser.add_argument("--dir_models", type=str, default='./models/trained_models/',
                        help="Directory that store trained models, default is './models/trained_models/'.")
    parser.add_argument("--dir_base_models", type=str, default='../workbench/_base_model/',
                        help="Directory that store base models, default is '../workbench/_base_model/'.")
    parser.add_argument("--dir_oracle_models", type=str, default='./models/oracle_models/',
                        help="Directory that store oracle models, default is './models/oracle_models/'.")
    parser.add_argument("--dir_pre_trained_models", type=str, default='./models/pre-trained_models/',
                        help="Directory that store pre-trained ESM models, default is './models/pre-trained_models/'.")
    parser.add_argument("--dir_exp", type=str, default='exp_{}/',
                        help="Directory to store experiment results.")

    parser.add_argument("--plots_name", type=str, default='exp{}_iter{}_game{}_rounds{}_eq{}_alg{}',
                        help="Name of the performance plots.")
    parser.add_argument("--dataset", type=str, default='gb1_4',
                        help="Type of the dataset.")
    parser.add_argument("--dataset_name", type=str, default='gb1_data.csv',
                        help="Name of the dataset file.")
    parser.add_argument("--esm_model_name", type=str, default='esm-1v',
                        help="Name of the ESM embedding model to use.")
    parser.add_argument("--is_plot_avg", type=bool, default=False,
                        help="Whether to plot average of multiple experiments or individual results, default is False.")
    parser.add_argument("--is_compare_opt", action='store_true',
                        help="Whether to compare with the optimal ucb strategy in each iteration or not, default is True.")
    parser.add_argument("--cuda", type=str, default='0',
                        help="cuda index, default is '0'.")
    parser.add_argument("--run_mode", type=str, default='default',
                        help="whether to run experiments or analyze the models, default is 'default (run experiments)'."
                             "options: [default, analyze, default_parallel_games]")
    parser.add_argument("--esm_embed_name", type=str, default='esm-1v', choices=['', 'esm-1v', 'esm-2'],
                        help="The name of the ESMEmbedding model to use"
                             "default is 'esm-1v'.")
    parser.add_argument("--nb_workers", type=int, default=4,
                        help="The number of workers to assign multiprocessing pool, default is 4.")
    return parser.parse_args()



