from apscheduler.schedulers.blocking import BlockingScheduler
import logging

from .auto import auto_update, test_func
from casestudy.models import Region

logging.basicConfig(filename='/tmp/log', level=logging.DEBUG)

sched = BlockingScheduler()

@sched.scheduled_job('cron', hour=12, minute=0)
def update_job():
    auto_update()

@sched.scheduled_job('interval', mintues=5)
def update_job():
    auto_update()

@sched.scheduled_job('interval', seconds=10)
def timed_job():
    print('This job is run every three minutes.')
    test_func()