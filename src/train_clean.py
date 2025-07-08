import os
# Disable OpenTelemetry before importing anything else
os.environ['OTEL_SDK_DISABLED'] = 'true'

import warnings
warnings.filterwarnings('ignore', message='.*OpenTelemetry.*')
warnings.filterwarnings('ignore', message='.*localhost:4318.*')

# Now import and run the main training function
from .train import main

if __name__ == "__main__":
    main()