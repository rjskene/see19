from datetime import datetime as dt
import numpy as np
import pandas as pd

import functools
import logging

import email, smtplib, ssl
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from git import Repo, remote
import nbformat as nbf
from nbconvert import MarkdownExporter
from nbconvert.writers import FilesWriter

from decouple import config

def update_readme(note=''):
    """
    Updates README.ipynb then converts to markdown
    """
    if config('HEROKU', cast=bool):
        SEE19PATH = config('ROOTPATH') + 'see19repo/'
    else:
        SEE19PATH = config('ROOTPATH') + 'casestudy/see19/'
    
    README_FILE = SEE19PATH + 'README.ipynb'
    README_NAME = SEE19PATH + 'README'

    nb = nbf.read(README_FILE, as_version=4)
    update_date = dt.now().strftime('%B %d, %Y')
    leadcell = {
        'cell_type': 'markdown',
       'metadata': {},
       'source': """# see19\n\n**An aggregation dataset and interface for visualizing 
           and analyzing Coronavirus Disease 2019 aka COVID19 
           aka C19**\n\n*Dataset Last Updated {}*{}
       """.format(update_date, note)
    }
    nb.cells.insert(0, leadcell)
    del nb.cells[1]

    # For some reason Node object is held as JSON "in memory"
    # so must convert back to NotebookNode
    # https://github.com/jupyter/nbconvert/issues/85
    nb = nbf.v4.to_notebook(nb)
    nbf.write(nb, README_FILE)
    
    readme = nbf.read(README_FILE, as_version=4)
    exporter = MarkdownExporter()
    (body, resources) = exporter.from_notebook_node(readme)

    write_file = FilesWriter()
    write_file.write(
        output=body,
        resources=resources,
        notebook_name=README_NAME
    )

def git_push(style='dataset'):
    if config('HEROKU', cast=bool):
        SEE19PATH = config('ROOTPATH') + 'see19repo/'
        repo = Repo(config('ROOTPATH') + 'see19repo/')
    else:
        SEE19PATH = config('ROOTPATH') + 'casestudy/see19/'
        repo = Repo(config('ROOTPATH'))
        
    assert style in ['dataset', 'testset', 'rm_only']
    
    adds = [SEE19PATH + item for item in ['dataset/', 'latest_dataset.txt', 'README.md']]
    m = 'update dataset'
    if style == 'testset':
        adds += [SEE19PATH + item for item in ['testset', 'latest_testset.txt']]
        m += ' and testset'
    elif style == 'rm_only':
        adds = SEE19PATH + 'README.md'
        m = 'update README'
    
    if style in ['dataset', 'testset']:
        m += ' {}'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'))

    repo.git.add(*adds)
    repo.git.commit(m=m)

    # if on HEROKU, the see19 repo IS the master
    # if local, see19 is the subtree
    if not config('HEROKU'):
        repo.git.subtree('push', 'see19', 'master', prefix='casestudy/see19/')
    
    repo.remote(name='origin').push()

def log_email(filename, critical=False):
    if critical:
        subject = 'CRITICAL ERROR'
    else:
        subject = '{} Update Log for see19'.format(dt.now().strftime('%B %d, %Y'))
        sender_email = 'covidchart@gmail.com'
        receiver_email = 'covidchart@gmail.com'
        password = config('EMAIL_PWORD')

    # Create a multipart message and set headers
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = receiver_email
    message['Subject'] = subject

    # Open PDF file in binary mode
    with open(filename, 'rb') as attachment:
        # Add file as application/octet-stream
        # Email client can usually download this automatically as attachment
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())

    # Encode file in ASCII characters to send by email    
    encoders.encode_base64(part)

    # Add header as key/value pair to attachment part
    part.add_header(
        'Content-Disposition',
        f'attachment; filename= {filename}',
    )

    # Add attachment to message and convert message to string
    message.attach(part)
    text = message.as_string()

    # Log in to server using secure context and send email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, text)

class ExceptionLogger:
    """
    Class for setting up attributes of log state, creating log file and adding logs
    """
    def __init__(self, logfile, level=None):
        
        # Required hack to configure handler
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
    
        self.level = level if level else logging.ERROR
    
        logging.basicConfig(filename=logfile, filemode='w', level=logging.ERROR)
    
    def wrap(self, level='exception'):
        """
        Decorator for catching and logging exceptions for whatever function is called
        Applied using traditional syntax of func = exc_wrap(level)(func)
        """
        assert level in ['exception', 'critical']

        def decorator(func):                
            @functools.wraps(func)
            def log_exception(*args, **kwargs):

                try:
                    print ('Running {}...'.format(func.__name__))
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    print (repr(e))
                    if level == 'exception':
                        logging.exception(func.__name__ + str(e))
                    elif level == 'critical':
                        logging.critical(func.__name__ + str(e))
            return log_exception
        return decorator

# TEST FUNCTIONS
def test_region_consistency(baseframe):
    # TEST REGION IDS, NAMES, AND CODES ONLY APPEAR IN ONE GROUPING
    
    cols = ['region_id', 'region_code', 'region_name']
    regarr = np.array([row.iloc[0].values for i, row in baseframe[cols].groupby(cols)])
    unique, counts = np.unique(regarr.T[0], return_counts=True)
    
    try:
        assert all(counts == 1)
    except Exception as e:
        raise Exception(' ...Failing Regions: ' + str(unique[counts > 1])) from e

def test_notnas(baseframe, count_type):
    """
    Test that case/death/test data is not null for certain major regions excluding mainland China
    Exception is augmented to include IDs of failing regions; makes log more convenient
    """
    countries = ['CAD', 'BRA', 'ITA', 'AUS', 'USA', 'IND', 'SPA', 'SWE', 'DEU', 'FIN', 'NOR', 'TWN', 'MYS', 'JPN', 'SGP', 'KOR', 'MEX']
    region_names = ['France', 'United Kingdom', 'Denmark']
    keyframe = baseframe[baseframe.country_code.isin(countries) | baseframe.region_name.isin(region_names)]

    na_groups = [df_group for region_id, df_group in keyframe.groupby('region_id') if df_group[count_type].isna().all()]
    
    try:
        assert len(na_groups) == 0 
    except Exception as e:
        df_na = pd.concat(na_groups)
        raise Exception(' ...Failing Regions: ' + str(df_na.region_id.unique())) from e
    
def test_duplicate_dates(casestudy):
    cols = ['region_id', 'date']
    df_test = casestudy.df[cols].copy(deep=True)
    df_test.date = pd.to_datetime(df_test.date).astype(np.int64)
    regarr = np.array([row.iloc[0].values for i, row in df_test.groupby(cols)])
    unique, counts = np.unique(regarr, axis=0, return_counts=True)

    try:
        assert all(counts == 1)
    except Exception as e:
        raise Exception(' ...Failing Regions: ' + str(df_days.region_id.unique())) from e

def test_duplicate_days(casestudy):
    cols = ['region_id', 'days']
    df_test = casestudy.df[cols].copy(deep=True)
    df_test.days = df_test.days.dt.days
    regarr = np.array([row.iloc[0].values for i, row in df_test.groupby(cols)])
    unique, counts = np.unique(regarr, axis=0, return_counts=True)    

    try:
        assert all(counts == 1)
    except Exception as e:
        dup = unique[counts > 1]
        raise Exception(' ...Failing Regions: ' + ' '.join(dup)) from e

def test_negative_days(casestudy):
    df_days = casestudy.df[casestudy.df.days.dt.days < 0]
    
    try:
        assert df_days.empty
    except Exception as e:
        regs_w_neg_days = df_days.region_id.unique().tolist()
        raise Exception(' ...Failing Regions: ' + ' '.join(regs_w_neg_days)) from e