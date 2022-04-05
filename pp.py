import settings
from prepare_data import prepdata

data_path=settings.DATA_PATH

prepdata(data_path=data_path, augmentation=settings.AUGMENT)