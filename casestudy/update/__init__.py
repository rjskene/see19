from datetime import datetime as dt

from decouple import config
from see19 import CaseStudy

def auto_update():
    from .baseframe import make_baseframe
    from .funcs import update_funcs
    from .helpers import test_region_consistency, test_notnas, test_duplicate_dates, test_duplicate_days, test_negative_days, log_email, git_push, update_readme, ExceptionLogger

    LOG_PATH = config('LOGPATH')
    filename = 'update-{}.log'.format(dt.now().strftime('%Y-%m-%d'))
    logfile = LOG_PATH + filename

    # Instantiate a new logger
    exc_logger = ExceptionLogger(logfile)

    # Loop through update functions and log any errors 
    for func in update_funcs:
        timeout = 10*60 if func.__name__ == 'update_msmts' else 60
        wrapfunc = exc_logger.wrap('exception', timeout=timeout)(func)
        wrapfunc(create=True)

    ### Test ###
    make_baseframe = exc_logger.wrap('critical')(make_baseframe)
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

    ### Send email and, If no critical errors, push to git 
    with open(logfile, 'r') as f:
        log_text = f.read()
    
    if 'CRITICAL' in log_text:
        log_email(logfile, critical=True)
    else:
        baseframe = make_baseframe(save=True)
        note = ''
        update_readme(note)
        git_push()
        log_email(logfile)