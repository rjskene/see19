from apscheduler.schedulers.blocking import BlockingScheduler
import logging

from casestudy.update import auto as auto_update
from casestudy.models import Region

logging.basicConfig(filename='/tmp/log', level=logging.DEBUG)

sched = BlockingScheduler()

@sched.scheduled_job('cron', hour=21, minute=9, second=20)
def update_job():
    print ('h?????')
    auto_update()

# @sched.scheduled_job('interval', seconds=3)
# def update_job():
#     print ('every 5 minutues')
    # auto_update()