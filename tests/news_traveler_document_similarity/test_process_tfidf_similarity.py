from news_traveler_document_similarity.tfidf_similarity import process_tfidf_similarity


def test_process_tfidf_similarity():
    base_document = "This summer, the 235-foot research vessel Marcus G. Langseth set out into the ocean off the Pacific Northwest. Trailing the ship were four electronic serpents, each five miles in length. These cables were adorned with scientific instruments able to peer into the beating heart of a monster a mile below the waves: Axial Seamount, a volcanic mountain."
    documents = [
        "In a study published this month in Scientific Reports, volcanologists reported using a novel technique to map out 58 square miles of Piton de la Fournaise’s shadowy underworld. Their survey revealed a 3D view of its insides, from the plumbing network of superheated hydrothermal fluids to scores of faults that allow magma to sneak up to the surface during eruptions.",
        "SKAFTAFELL, Iceland — Just north of here, on the far side of the impenetrable Vatnajokull ice sheet, lava is spewing from a crack in the earth on the flanks of Bardarbunga, one of Iceland’s largest volcanoes. By volcanologists’ standards, it is a peaceful eruption, the lava merely spreading across the landscape as gases bubble out of it. For now, those gases — especially sulfur dioxide, which can cause respiratory and other problems — are the main concern, prompting health advisories in the capital, Reykjavik, 150 miles to the west, and elsewhere around the country.",
        "After its eruption on May 3, the Kilauea volcano, located on Hawaii’s Big Island, has continued to erupt. As the volcano’s lava lake drained back into the ground, scientists warned that a major eruption was bound to happen. After all, this is not the first time this has occurred.",
        "Whether or not dogs dream isn’t known with scientific certainty, but it sure is difficult to imagine that they don’t. We’ve all watched our dogs demonstrate behaviors in their sleep that resemble what they do in a fully awake state. Paddling legs, whining, growling, wagging tails, chewing jowls, and twitching noses inspire us to wonder what our dogs are dreaming about.",
        "Located in Oakland, California, The Cat Town Café & Adoption Center has become the first cat café in the United States which opened in late October, reports Carol Pogash of the New York Times. Inside the café you won’t see cages, but free roaming cats looking for a forever home.",
    ]
    select_count = 3

    result = process_tfidf_similarity(base_document, documents, select_count)

    assert len(result) == 3
