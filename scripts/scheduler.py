from apscheduler.schedulers.blocking import BlockingScheduler
import logging

from casestudy.update import pull, test, push
from casestudy.models import Region

logging.basicConfig(filename='/tmp/log', level=logging.DEBUG)

sched = BlockingScheduler()

@sched.scheduled_job('cron', hour=7, minute=13, second=0)
def pull_job():
    pull()

@sched.scheduled_job('cron', hour=7, minute=17, second=0)
def test_job():
    test()

@sched.scheduled_job('cron', hour=7, minute=21, second=0)
def push_job():
    push()