from apscheduler.schedulers.blocking import BlockingScheduler
import logging

from casestudy.update import pull, test, push
from casestudy.models import Region

logging.basicConfig(filename='/tmp/log', level=logging.DEBUG)

sched = BlockingScheduler()

@sched.scheduled_job('cron', hour=13, minute=0, second=0)
def pull_job():
    auto()

# @sched.scheduled_job('cron', hour=11, minute=15, second=0)
# def test_job():
#     test()

# @sched.scheduled_job('cron', hour=11, minute=25, second=0)
# def push_job():
#     push()