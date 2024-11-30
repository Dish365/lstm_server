from .base import *
import os

DEBUG = True
SECRET_KEY = 'django-insecure-)efm!+b!#)cu8_z#sb25m15cz#%2@m5334v%5@h1t(lp)^l-(e'

ALLOWED_HOSTS = ['localhost', '127.0.0.1']
CORS_ALLOW_ALL_ORIGINS = True

# Model files directory for local development
MODELS_DIR = os.path.join(BASE_DIR, 'models') 