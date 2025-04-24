"""#from celery import Celery
import os

# Set up Celery
def make_celery(app):
    celery = Celery(
        app.import_name,
        backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL']
    )
    celery.conf.update(app.config)
    return celery

# Initialize Celery
celery = make_celery(app)  # Assuming `app` is initialized in `main.py`
"""