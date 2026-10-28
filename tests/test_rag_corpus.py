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


PAGE = {"title": "Leaf rot", "url": "u", "classes": ["Tomato___Leaf_Mold"], "text": (
    "Leaf rot is a disease.\n\n\n"
    "== Symptoms ==\nBrown spots.\n\nYellow leaves.\n\n\n"
    "=== On fruit ===\nSoft fruit.\n\n\n"
    "== Management ==\nRotate crops. Remove debris.\n\n\n"
    "== References ==\nSmith 2001.\n\n\n"
    "=== Sources ===\nA book.\n\n\n"
    "== External links ==\nhttp://example.org\n")}


def test_sections_keep_headings_and_drop_references_with_their_subsections():
    from leafscan.rag.corpus import sections

    secs = sections(PAGE)
    assert [s["path"] for s in secs] == [["Leaf rot"], ["Leaf rot", "Symptoms"],
                                         ["Leaf rot", "Symptoms", "On fruit"],
                                         ["Leaf rot", "Management"]]
    assert "Smith" not in " ".join(s["text"] for s in secs)
    assert "A book" not in " ".join(s["text"] for s in secs)


def test_chunks_respect_the_word_limit_and_lose_no_text():
    from leafscan.rag.corpus import chunk_text

    paragraphs = [" ".join(f"w{p}_{i}" for i in range(7)) + "." for p in range(10)]
    long_paragraph = " ".join(f"Sentence {i} has five words." for i in range(10))
    text = "\n\n".join(paragraphs + [long_paragraph])
    chunks = chunk_text(text, max_words=20)
    assert all(len(c.split()) <= 20 for c in chunks)
    assert " ".join(chunks).split() == text.split()


def test_chunk_pages_never_cross_sections():
    from leafscan.rag.corpus import chunk_pages, stats

    chunks = chunk_pages([PAGE], max_words=200)
    assert [c["text"] for c in chunks] == ["Leaf rot is a disease.", "Brown spots. Yellow leaves.",
                                           "Soft fruit.", "Rotate crops. Remove debris."]
    assert chunks[3]["section"] == "Leaf rot > Management"
    assert len({c["id"] for c in chunks}) == 4
    assert stats([PAGE])["per_class"]["Tomato___Leaf_Mold"]["has_management_section"]
