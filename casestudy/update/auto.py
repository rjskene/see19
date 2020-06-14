from datetime import datetime as dt

from git import Repo
from decouple import config
from see19 import CaseStudy

def test_func():
    print ('this is a test_func')

def auto(test=False):
    from .funcs import update_funcs
    from .helpers import test_region_consistency, test_notnas, test_duplicate_dates, test_duplicate_days, test_negative_days, log_email, git_push, update_readme, ExceptionLogger

    from .baseframe import make

    print ('this is live!')
    LOG_PATH = config('ROOTPATH') + 'casestudy/update/update_logs/'
    filename = 'update-{}.log'.format(dt.now().strftime('%Y-%m-%d'))
    logfile = LOG_PATH + filename

    print ('logger')
    # Instantiate a new logger
    exc_logger = ExceptionLogger(logfile)

    #### IF ON HEROKU HAVE TO GIT CLONE THE REPO ###
    # Do this first, b/c if not possible, the rest of the code is useless
    if config('HEROKU', cast=bool):
        wrapfunc = exc_logger.wrap('critical')(Repo.clone_from)
        wrapfunc(config('SEE19GITURL'), '/app/see19repo/')
        print ('cloned repo')
    
    # Loop through update functions and log any errors 
    for func in update_funcs:
        print ('udpating ' + func.__name__)
        wrapfunc = exc_logger.wrap('exception')(func)
        wrapfunc(create=True)

    ### Test ###
    print ('making baseframe')
    make_baseframe = exc_logger.wrap('critical')(make)
    baseframe = make_baseframe()
    
    test_region_consistency = exc_logger.wrap('critical')(test_region_consistency)
    test_region_consistency(baseframe)

    for count_type in CaseStudy.COUNT_TYPES:
        test_notnas = exc_logger.wrap('exception')(test_notnas)
        test_notnas(baseframe, count_type)

    factors_with_dmas = ['strindex']
    kwargs = {'factors': CaseStudy.ALL_FACTORS, 'interpolate_method': {'method': 'linear'}}
    casestudy = CaseStudy(baseframe, **kwargs)
    
    test_duplicate_dates = exc_logger.wrap('exception')(test_duplicate_dates)
    test_duplicate_dates(casestudy)
    test_duplicate_days = exc_logger.wrap('exception')(test_duplicate_days)
    test_duplicate_days(casestudy)
    test_negative_days = exc_logger.wrap('exception')(test_negative_days)
    test_negative_days(casestudy)
    
    print ('get here')
    ### Send email and, If no critical errors, push to git 
    with open(logfile, 'r') as f:
        log_text = f.read()
    
    if 'CRITICAL' in log_text:
        log_email(logfile, critical=True)
    else:
        baseframe = make(save=True)
        note = ''
        update_readme(note)

        if not test:
            print ('push to git')
            git_push()
            print ('send log email')
            log_email(logfile)

    print ('UPDATE COMPLETE')