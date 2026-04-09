import yaml

def read_config(path_yaml):
    """Can't seem to remember the best syntax for reading
    in YAML files, so here we are.

    Args:
        path_yaml (str): Path to the YAML config file.
    """
    try:
        with open(path_yaml, 'r') as yaml_file:
            config_file = yaml.safe_load(yaml_file)
        print("Read " + path_yaml + " config file")
        return config_file
    except Exception as e:
        if isinstance(path_yaml, dict):
            raise TypeError(f"""path_yaml is not a file path; 
                            type(path_yaml) -> {type(path_yaml)}""")
        raise
