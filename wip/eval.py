from thefuzz import process
from utils import create_presidio_engine, load_full_files
from presidio_anonymizer import AnonymizerEngine


engine = create_presidio_engine()
anonymizer = AnonymizerEngine()

def do_analysis(text: list, language: str):
    found = 0
    for s in text[:150]:
        results = engine.analyze(text=s, language=language)
        anonymized = anonymizer.anonymize(
            text=s,
            analyzer_results=results
        )
        hyp, _ = process.extractOne(anonymized.text, text[:150])
        if hyp == s:
            found += 1

    return found


def main():
    d = load_full_files()
    for k, v in d.items():
        f = do_analysis(v, str(k)[-2:])
        ratio = f/150
        print(f"Results for {k}: {ratio}")


if __name__ == "__main__":
    main()