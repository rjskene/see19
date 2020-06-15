from apscheduler.schedulers.blocking import BlockingScheduler
import logging

from casestudy.update import pull, test, push
from casestudy.models import Region

logging.basicConfig(filename='/tmp/log', level=logging.DEBUG)

sched = BlockingScheduler()

@sched.scheduled_job('cron', hour=23, minute=40, second=0)
def pull_job():
    pull()

@sched.scheduled_job('cron', hour=23, minute=50, second=0)
def test_job():
    test()

@sched.scheduled_job('cron', hour=23, minute=59, second=59)
def push_job():
    push()