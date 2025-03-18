import logging
import threading
import inspect

class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    RESET = "\033[0m"

class SingletonLogger:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize(*args, **kwargs)
        return cls._instance

    def _initialize(self, level=logging.DEBUG):
        self.logger = logging.getLogger()
        self.logger.setLevel(level)

        log_file_path = "./data/log_file.log"
        file_handler = logging.FileHandler(log_file_path, mode='w')
        file_handler.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        file_handler.setFormatter(self._get_file_formatter())
        console_handler.setFormatter(self._get_colored_formatter())

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _get_colored_formatter(self):
        class ColoredFormatter(logging.Formatter):
            def format(self, record):
                frame = inspect.stack()[8]  # Adjust index to reach the caller
                module = inspect.getmodule(frame[0])
                module_name = module.__name__ if module else ''
                class_name = ''
                if 'self' in frame[0].f_locals:
                    class_name = frame[0].f_locals['self'].__class__.__name__
                function_name = frame[3]
                caller_name = f"{module_name}.{class_name}.{function_name}".strip('.')

                # Apply color to the log level
                color = Colors.WHITE  # default to white
                if record.levelno == logging.DEBUG:
                    color = Colors.CYAN
                elif record.levelno == logging.INFO:
                    color = Colors.GREEN
                elif record.levelno == logging.WARNING:
                    color = Colors.YELLOW
                elif record.levelno == logging.ERROR:
                    color = Colors.RED
                elif record.levelno == logging.CRITICAL:
                    color = Colors.PURPLE

                record.levelname = f"{color}{record.levelname}{Colors.RESET}"
                record.name = caller_name
                return super().format(record)

        return ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def _get_file_formatter(self):
        return logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def set_level(self, level):
        self.logger.setLevel(level)

    def get_logger(self):
        return self.logger