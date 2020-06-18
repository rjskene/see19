web: gunicorn zooscraper.wsgi --log-file -
clock: python3 manage.py runscript scheduled_updates
tweet: python3 manage.py runscript tweet_scheduler