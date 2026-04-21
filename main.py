import argparse

from search import display_results, search


def main():
    parser = argparse.ArgumentParser(description="Film scene search")
    parser.add_argument("query", help="Natural language search query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--film-id", default=None, help="Filter by film ID")
    args = parser.parse_args()

    results = search(args.query, top_k=args.top_k, film_id=args.film_id)
    display_results(results)


if __name__ == "__main__":
    main()
