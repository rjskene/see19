from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor
from apscheduler.schedulers.blocking import BlockingScheduler
from casestudy.covidcharts import tweetledee

sched = BlockingScheduler(
    executors={
        'threadpool': ThreadPoolExecutor(max_workers=9),
        'processpool': ProcessPoolExecutor(max_workers=3)
        }
)

@sched.scheduled_job('cron', hour=5, minute=59, second=45)
def tweet_job():
    tweetledee()

def run():
    print ('initiating tweet scheduler')
    sched.start()