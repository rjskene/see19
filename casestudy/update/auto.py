from datetime import datetime as dt

import gc

from git import Repo
from decouple import config
from see19 import CaseStudy

def test_func():
    print ('this is a test_func')

def auto(test=False):
    from .funcs import update_funcs
    from .helpers import test_region_consistency, test_notnas, test_duplicate_dates, test_duplicate_days, test_negative_days, log_email, git_push, update_readme, ExceptionLogger

    from .baseframe import make

    LOG_PATH = config('ROOTPATH') + 'casestudy/update/update_logs/'
    filename = 'update-{}.log'.format(dt.now().strftime('%Y-%m-%d'))
    logfile = LOG_PATH + filename

    # Instantiate a new logger
    print ('Instantiate new exception logger')
    exc_logger = ExceptionLogger(logfile)

    #### IF ON HEROKU HAVE TO GIT CLONE THE REPO ###
    # Do this first, b/c if not possible, the rest of the code is useless
    if config('HEROKU', cast=bool):
        wrapfunc = exc_logger.wrap('critical')(Repo.clone_from)
        wrapfunc(config('SEE19GITURL'), '/app/see19repo/')
    
    # Loop through update functions and log any errors 
    for func in update_funcs:
        print ('udpating ' + func.__name__)
        wrapfunc = exc_logger.wrap('exception')(func)
        wrapfunc(create=True)
        # func(create=True)
        # from pympler import muppy, summary
        # all_objects = muppy.get_objects()
        # sum1 = summary.summarize(all_objects)
        # Prints out a summary of the large objects
        # summary.print_(sum1)
        # lists = [ao for ao in all_objects if isinstance(ao, list)]
        # print (lists)

    ### Test ###
    print ('making baseframe')
    make_baseframe = exc_logger.wrap('critical')(make)
    baseframe = make_baseframe()
    
    print ('Region Consistency test')
    test_region_consistency = exc_logger.wrap('critical')(test_region_consistency)
    test_region_consistency(baseframe)

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
    
    ### Send email and, If no critical errors, push to git 
    print ('Reading logfile...')
    with open(logfile, 'r') as f:
        log_text = f.read()
    
    if 'CRITICAL' in log_text:
        print ('There were critical errors. Dataset will not be updated.')
        log_email(logfile, critical=True)
    else:
        print ('No critical errors. Saving baseframe to disk.')
        baseframe = make(save=True)

        print ('Updating readme')
        note = ''
        update_readme(note)

        if not test:
            print ('push to git')
            git_push()
            print ('send log email')
            log_email(logfile)

    print ('UPDATE COMPLETE')