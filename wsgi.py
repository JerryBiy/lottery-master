from dlt_ai.settings import SETTINGS
from dlt_ai.webapp import create_app


application = create_app(start_scheduler=SETTINGS.scheduler_enabled)
