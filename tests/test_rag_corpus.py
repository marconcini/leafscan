from leafscan.rag.corpus import PAGES

DISEASES = {
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
    "Cherry_(including_sour)___Powdery_mildew",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot", "Pepper,_bell___Bacterial_spot", "Potato___Early_blight",
    "Potato___Late_blight", "Squash___Powdery_mildew", "Strawberry___Leaf_scorch",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
}


def test_every_disease_has_a_page_and_only_real_classes_are_used():
    covered = {c for classes in PAGES.values() for c in classes}
    assert covered == DISEASES
