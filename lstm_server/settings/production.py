from .base import *
import os

DEBUG = False
SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY')

# Domain settings
ALLOWED_HOSTS = [
    'fsroas.com',
    'www.fsroas.com',
    os.environ.get('ALLOWED_HOSTS', '').split(',')
]

# CORS settings
CORS_ALLOWED_ORIGINS = [
    'https://fsroas.com',
    'https://www.fsroas.com',
    *os.environ.get('CORS_ALLOWED_ORIGINS', '').split(',')
]
CORS_ALLOW_ALL_ORIGINS = False

# Security settings
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = 'DENY'
SECURE_HSTS_SECONDS = 31536000  # 1 year
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True

# Domain-specific settings
CSRF_TRUSTED_ORIGINS = [
    'https://fsroas.com',
    'https://www.fsroas.com',
]

# Static and media files
STATIC_URL = 'static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
MEDIA_URL = 'media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'mediafiles')

# Model files directory
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Logging configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'file': {
            'level': 'ERROR',
            'class': 'logging.FileHandler',
            'filename': os.path.join(BASE_DIR, 'logs', 'django.log'),
        },
    },
    'loggers': {
        'django': {
            'handlers': ['file'],
            'level': 'ERROR',
            'propagate': True,
        },
    },
} 