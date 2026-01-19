from lib.normalize import normalize_name, birth_year_from_date


def test_normalize_basic():
    assert normalize_name("Éclair d'Ô") == "ECLAIR D O"
    assert normalize_name("Cheval   Bleu!!") == "CHEVAL BLEU"


def test_birth_year():
    assert birth_year_from_date("2020-05-11") == 2020
    assert birth_year_from_date("11/05/2020") == 2020
    assert birth_year_from_date("") is None
