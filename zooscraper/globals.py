import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from django.db import transaction

from decouple import config

class ChromeInstantiator():
    # Context Manager for instantiating a Chrome browser for Selenium
    def __init__(self, headless=True, download_dir=None):
        self.download_dir = '/Users/spindicate/Downloads/' if download_dir is None else download_dir
        self.headless = headless

        if not isinstance(headless, bool):
            raise AttributeError('headless must be boolean')
        else:
            self.headless = headless

        if self.headless:
            # Instantiate the remote WebDriver
            self.options = Options()
            self.options.add_argument('--no-sandbox')
            self.options.set_headless(headless=True)
            self.options.binary_location = config('GOOGLE_CHROME_BIN')
        else:
            self.options=None

    def enable_download_in_headless_chrome(self, download_dir=None):
        # Via https://github.com/shawnbutton/PythonHeadlessChrome/blob/master/driver_builder.py
        """
        there is currently a "feature" in chrome where
        headless does not allow file download: https://bugs.chromium.org/p/chromium/issues/detail?id=696481
        This method is a hacky work-around until the official chromedriver support for this.
        Requires chrome version 62.0.3196.0 or above.
        """
        download_dir = download_dir if download_dir else self.download_dir
        # add missing support for chrome "send_command"  to selenium webdriver
        self.chrome.command_executor._commands["send_command"] = ("POST", '/session/$sessionId/chromium/send_command')

        params = {'cmd': 'Page.setDownloadBehavior', 'params': {'behavior': 'allow', 'downloadPath': download_dir}}
        command_result = self.chrome.execute("send_command", params)
        # for key in command_result:
        #     print("result:" + key + ":" + str(command_result[key]))

    def __enter__(self):
        print ('Instantiating Chrome WebDriver...')
        self.chrome = webdriver.Chrome(executable_path=config('CHROMEDRIVER_PATH'),
                    options=self.options
        )
        self.enable_download_in_headless_chrome()
        
        return self.chrome

    def __exit__(self, *args):
        time.sleep(10)
        self.chrome.quit()

def max_bulk_create(objs):
    """Wrapper to overcome the size limitation of standard bulk_create()"""
    model = type(objs[0])
    if objs:
        BULK_SIZE = int(900/len(model._meta.fields))
        with transaction.atomic():
            for i in range(0, len(objs), BULK_SIZE):
                model.objects.bulk_create(objs[i: i + BULK_SIZE])