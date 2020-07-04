from apscheduler.schedulers.blocking import BlockingScheduler
import logging

from casestudy.update import auto, pull, test, push
from casestudy.models import Region

logging.basicConfig()
logging.getLogger('apscheduler').setLevel(logging.DEBUG)

sched = BlockingScheduler()

@sched.scheduled_job('cron', hour=12, minute=15, second=0)
def update_job():
    auto()

def run():
    print ('initiating scheduler')
    sched.start()