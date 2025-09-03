class ParametricDisk:
    def __init__(self, config):
        self.params_file = read_config(config)
        self.params_init = self._get_initial_params()
        self._load_dirs()
        self._load_metadata()
        self._load_klparams()
        self.klbasis = self._loadbasis()