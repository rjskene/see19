from datetime import datetime as dt

import gc
from git import Repo
from decouple import config
from see19 import CaseStudy

LOGS_PATH = config('ROOTPATH') + 'casestudy/update/logs/'
MODELLOGS_PATH = config('ROOTPATH') + 'casestudy/update/logs/models/'
TESTLOGS_PATH = config('ROOTPATH') + 'casestudy/update/logs/tests/'

def auto():
    print ('inside auto')
    pull()
    test()
    push()
    print ('end auto')

def merge_logs(filename):
    """
    Record critical error if model or test log is not found
    """
    try:
        with open(MODELLOGS_PATH + filename, 'r') as f:
            lines = f.readlines()
    except:
        lines = ['CRITICAL:root:CANNOT READ MODEL LOG']
    try:
        with open(TESTLOGS_PATH + filename, 'r') as f:
            lines += f.readlines()
    except:
        lines += ['CRITICAL:root:CANNOT READ TEST LOG']

    with open(LOGS_PATH + filename, 'w') as f:
        f.writelines(lines)

def pull(test=False):
    from .funcs import update_funcs
    from .helpers import ExceptionLogger

    # Instantiate a new logger
    print ('Instantiate new exception logger')
    today = dt.now().strftime('%Y-%m-%d')
    filename = '{}.log'.format(today)
    logfile = MODELLOGS_PATH + filename
    exc_logger = ExceptionLogger(logfile)
    
    # Loop through the update functions and log any errors 
    for func in update_funcs:
        print ('Running {}'.format(func.__name__))
        wrapfunc = exc_logger.wrap('exception')(func)
        wrapfunc(create=True)

def test(): 
    from .helpers import ExceptionLogger, test_region_consistency, test_notnas, test_duplicate_dates, test_duplicate_days, test_negative_days, test_data_is_timely
    from .baseframe import make
    """
    """
    # Instantiate a new logger
    print ('Instantiate new exception logger')
    today = dt.now().strftime('%Y-%m-%d')
    filename = '{}.log'.format(today)
    logfile = TESTLOGS_PATH + filename
    exc_logger = ExceptionLogger(logfile)

    print ('making baseframe')
    make_baseframe = exc_logger.wrap('critical')(make)
    baseframe = make_baseframe()
    
    print ('Region Consistency test')
    test_region_consistency = exc_logger.wrap('critical')(test_region_consistency)
    test_region_consistency(baseframe)

    print ('Test data is timely for each region')
    test_data_is_timely = exc_logger.wrap('exception')(test_data_is_timely)
    test_data_is_timely(baseframe)

    for count_type in CaseStudy.COUNT_TYPES:
        print ('Not Na test for {}'.format(count_type))
        test_notnas = exc_logger.wrap('exception')(test_notnas)
        test_notnas(baseframe, count_type)

    factors_with_dmas = ['strindex']
    kwargs = {'factors': CaseStudy.ALL_FACTORS, 'interpolate_method': {'method': 'linear'}}
    casestudy = CaseStudy(baseframe, **kwargs)
    
    print ('Casestudy tests...')
    print ('Test duplicate dates')
    test_duplicate_dates = exc_logger.wrap('exception')(test_duplicate_dates)
    test_duplicate_dates(casestudy)
    
    print ('Test duplicate days')
    test_duplicate_days = exc_logger.wrap('exception')(test_duplicate_days)
    test_duplicate_days(casestudy)

    print ('Test negative days')
    test_negative_days = exc_logger.wrap('exception')(test_negative_days)
    test_negative_days(casestudy)

def push(test=False):
    from .baseframe import make
    from .helpers import git_push, log_email, update_readme

    ### Merge logs, check for critical errors, push to git, and email log
    print ('Merge logs...')
    today = dt.now().strftime('%Y-%m-%d')
    filename = '{}.log'.format(today)
    merge_logs(filename)

    print ('Reading merge log file...')
    with open(LOGS_PATH + filename, 'r') as f:
        log_text = f.read()
    
    # if 'CRITICAL' in log_text:
    if False:
        print ('There were critical errors. Dataset will not be updated.')
        log_email(LOGS_PATH + filename, critical=True)
    else:
        #### IF ON HEROKU HAVE TO GIT CLONE THE REPO ###
        if config('HEROKU', cast=bool):
            print ('Cloning the see19 repo ...')
            Repo.clone_from(config('SEE19GITURL'), config('ROOTPATH') + 'see19repo/')
        
        print ('No critical errors. Saving baseframe to disk.')
        baseframe = make(save=True)

        print ('Updating readme')
        note = ''
        update_readme(note)

        if not test:
            print ('push to git')
            git_push()
            print ('send log email')
            log_email(LOGS_PATH + filename)

    print ('UPDATE COMPLETE')