def generate_recommendations(profile, risk, prediction, shap_factors=None):
    recs = []
    bmi = profile.get('bmi')
    smoker = int(profile.get('smoker') or 0)
    diabetes = int(profile.get('diabetes') or 0)
    bp = profile.get('bloodpressure')
    exercise = int(profile.get('regular_ex') or 0)
    hereditary = str(profile.get('hereditary_diseases') or '')

    if smoker:
        recs.append('Consider a structured smoking-cessation program or support service.')
    if bmi is not None and bmi >= 30:
        recs.append('Consider gradual, sustainable weight-management habits and appropriate physical activity.')
    if bp is not None and bp >= 140:
        recs.append('Consider discussing repeated high blood-pressure readings with a qualified healthcare professional.')
    if diabetes:
        recs.append('Maintain regular diabetes monitoring and follow advice from your healthcare professional.')
    if not exercise:
        recs.append('If appropriate for you, gradually increase regular physical activity.')
    if hereditary and hereditary.lower() not in {'nodisease', 'none', 'nan'}:
        recs.append('Keep relevant family/medical history available when discussing preventive care with a professional.')
    if not recs:
        recs.append('Continue your current healthy habits and routine preventive care.')

    recs.append('These are general wellness suggestions and do not guarantee a lower insurance premium or claim amount.')
    return recs
