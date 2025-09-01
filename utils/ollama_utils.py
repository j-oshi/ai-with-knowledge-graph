import ollama

def list_of_installed_models():
    """
    List of available ollama models

    Returns:
        list: A list of model names.
    """
    models_info = ollama.list()
    model_names = tuple(model["model"] for model in models_info["models"])
    return model_names

def check_if_model_exist(model: str):
    model_list = list_of_installed_models()
    if model in model_list:
        return True
    print(f'{model} is not installed.')
    return False