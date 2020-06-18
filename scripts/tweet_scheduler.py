from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor
from apscheduler.schedulers.blocking import BlockingScheduler
from casestudy.covidcharts import tweetledee

logging.basicConfig()
logging.getLogger('apscheduler').setLevel(logging.DEBUG)

sched = BlockingScheduler(
    executors={
        'threadpool': ThreadPoolExecutor(max_workers=9),
        'processpool': ProcessPoolExecutor(max_workers=3)
        }
)

@sched.scheduled_job('cron', hour=6, minute=25, second=0)
def tweet_job():
    tweetledee()

def run():
    print ('initiating tweet scheduler')
    sched.start()