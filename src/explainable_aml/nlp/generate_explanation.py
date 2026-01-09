from typing import Dict, Any

def generate_nlp_explanation(explanation: Dict[str, Any], transaction_features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a plain-English explanation from SHAP attributions.

    Args:
        explanation (dict): From explain_transaction
        transaction_features (dict): Original features

    Returns:
        dict: Structured explanation with text + metadata
    """
    risk_score = explanation["risk_score"]
    alert_flag = explanation["alert_flag"]
    risk_band = explanation.get("risk_band", "Unknown")
    top_features = explanation["top_features"]
    ood_flag = explanation.get("ood_flag", False)
    ood_features = explanation.get("ood_features", [])

    # ---- Explanation header ----
    explanation_text = (
        f"This transaction received a model risk score of {risk_score:.4f} ({risk_band} risk band). "
    )

    if alert_flag:
        explanation_text += (
            "Based on the configured threshold, the model flagged this transaction as higher risk. "
        )
    else:
        explanation_text += (
            "Based on the configured threshold, the model did not flag this transaction. "
        )

    # ---- OOD Warning ----
    if ood_flag:
        explanation_text += (
            f"Warning: The following features have values outside the range observed during training: {', '.join(ood_features)}. "
            "As a result, the model's assessment may be unreliable. "
        )

    explanation_text += "The primary factors influencing this score were:\n"

    # ---- Build explanation pattern ID ----
    pattern_parts = ["ALERT" if alert_flag else "NO_ALERT"]

    # ---- Check for all-zero contributions ----
    all_zero = all(abs(item["contribution"]) < 1e-6 for item in top_features)

    if all_zero:
        explanation_text += (
            "The model assigned a low risk score, but none of the input features materially influenced this result. "
            "This typically occurs when feature values fall outside the range observed during training, "
            "limiting the model’s ability to assess risk reliably.\n"
        )
    else:
        explanation_text += "The primary factors influencing this score were:\n"

    # ---- Feature-level explanations ----
    features_used = []

    for item in top_features:
        feature = item["feature"]
        contribution = item["contribution"]
        value = transaction_features.get(feature, "unknown")

        if contribution > 0:
            direction = "increased"
            sign = "POS"
        elif contribution < 0:
            direction = "decreased"
            sign = "NEG"
        else:
            direction = "did not materially affect"
            sign = "ZERO"

        pattern_parts.append(f"{sign}_{feature.upper()}")

        # Defensive value formatting
        if isinstance(value, (int, float)):
            value_str = f"{value:.2f}"
        else:
            value_str = str(value)

        if not all_zero:
            explanation_text += (
                f"- The feature '{feature.replace('_', ' ')}' "
                f"with a value of {value_str} {direction} the model’s risk score "
                f"by {abs(contribution):.4f}.\n"
            )

        features_used.append(feature)

    pattern_id = "_".join(pattern_parts)

    return {
        "text": explanation_text,
        "pattern_id": pattern_id,
        "features_used": features_used,
        "generator_version": "nlp_v1.0.0",
    }
