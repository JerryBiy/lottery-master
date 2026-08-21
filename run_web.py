from dlt_ai.webapp import create_app
from dlt_ai.settings import SETTINGS


app = create_app(start_scheduler=SETTINGS.scheduler_enabled)


if __name__ == "__main__":
    if SETTINGS.debug:
        app.run(host=SETTINGS.host, port=SETTINGS.port, debug=True, use_reloader=False)
    else:
        from waitress import serve

        serve(app, host=SETTINGS.host, port=SETTINGS.port, threads=SETTINGS.threads)
