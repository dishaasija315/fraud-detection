import pandas as pd


def create_features(data, feature_order):
    """
    Takes raw transaction data (dict, Pydantic model, or DataFrame) and
    engineers the features: errorBalanceOrig, errorBalanceDest, hour, and is_transfer.
    Returns a DataFrame with columns in the exact order saved in models/model_meta.json.
    """
    if hasattr(data, "model_dump"):
        data_dict = data.model_dump()
    elif hasattr(data, "dict"):
        data_dict = data.dict()
    elif isinstance(data, dict):
        data_dict = data
    else:
        data_dict = None

    if data_dict is not None:
        df = pd.DataFrame([data_dict])
    else:
        df = pd.DataFrame(data).copy()

    # Engineer features matching notebooks/02_features.ipynb
    df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    df['hour'] = df['step'] % 24
    df['is_transfer'] = (df['type'] == 'TRANSFER').astype(int)

    # Return columns in exact order from model metadata
    return df[feature_order]
