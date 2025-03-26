from transformers import RobertaForSequenceClassification

def load_model(model_path):
    """
    Utility function to load the classification model.
    
    :param model_path: Path to the model directory
    :return: Loaded RoBERTa model for sequence classification
    """
    try:
        model = RobertaForSequenceClassification.from_pretrained(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading classification model: {e}")