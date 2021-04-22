import pickle
import os


class SettingManager(object):
    SAVE_PATH = "data/settings/"
    settings = []

    # 初始化
    def __init__(self):
        super(object, self).__init__()
        for file_name in os.listdir(self.SAVE_PATH):
            if file_name.endswith('.st'):
                fd = open(self.SAVE_PATH + file_name, 'rb')
                setting = pickle.load(fd)
                fd.close()
                self.settings.append(setting)

    def add(self, setting):
        self.settings.append(setting)
        self._save(setting)

    def modify(self, setting):
        for s in self.settings:
            if s[0] == setting[0]:
                s = setting
                self._save(setting)
                break

    def remove(self, setting):
        for s in self.settings:
            if s[0] == setting[0]:
                self.settings.remove(s)
                os.remove(self._get_file_path(setting))
                break

    def _save(self, setting):
        fd = open(self._get_file_path(setting), 'wb')
        pickle.dump(setting, fd, 2)
        fd.close()

    def _get_file_path(self, setting):
        return self.SAVE_PATH + setting[0] + ".st"
