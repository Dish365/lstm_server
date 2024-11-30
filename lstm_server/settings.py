print("BASE_DIR:", settings.BASE_DIR)
print("Model path:", os.path.join(settings.BASE_DIR, 'models', MODEL_FILENAME)) 

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'file': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': '/var/log/gunicorn/django.log',
        },
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'INFO',
    },
    'loggers': {
        'django': {
            'handlers': ['file'],
            'level': 'INFO',
            'propagate': True,
        },
        'api': {
            'handlers': ['file'],
            'level': 'DEBUG',
            'propagate': True,
        },
    },
} 