import settings
from prepare_data import prepdata


data_path=settings.TESTDATA_PATH

prepdata(data_path=data_path, augmentation=False)
