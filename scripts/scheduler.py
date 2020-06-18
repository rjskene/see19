from apscheduler.schedulers.blocking import BlockingScheduler
import logging

from casestudy.update import auto, pull, test, push
from casestudy.models import Region

logging.basicConfig(filename='/tmp/log', level=logging.DEBUG)

sched = BlockingScheduler()

@sched.scheduled_job('cron', hour=12, minute=30, second=0)
def update_job():
    auto()