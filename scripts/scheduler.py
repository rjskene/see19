from apscheduler.schedulers.blocking import BlockingScheduler
import logging

from casestudy.update import auto, pull, test, push
from casestudy.models import Region

logging.basicConfig(filename='/tmp/log', level=logging.DEBUG)

sched = BlockingScheduler()

@sched.scheduled_job('cron', hour=21, minute=12, second=0)
def pull_job():
    print ('begin job')
    auto()

@sched.scheduled_job('interval', seconds=5)
def timed_job():
    print('This job is run every 5 seconds.')

# @sched.scheduled_job('cron', hour=11, minute=15, second=0)
# def test_job():
#     test()

# @sched.scheduled_job('cron', hour=11, minute=25, second=0)
# def push_job():
#     push()