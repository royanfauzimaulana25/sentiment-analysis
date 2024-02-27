import tensorflow as tf
LABEL_KEY = "label"
FEATURE_KEY = "sentences"
def transformed_name(key):
    """Renaming transformed features"""
    # print(key)
    return key + "_xf"
    
def preprocessing_fn(inputs):
    """
    Preprocess input features into transformed features
    
    Args:
        inputs: map from feature keys to raw features.
    
    Return:
        outputs: map from feature keys to transformed features.    
    """
    
    outputs = {}
    # print(inputs)
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(inputs[FEATURE_KEY])
    
    return outputs
