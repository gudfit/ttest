from __future__ import annotations


def main():
    # Stage3 for PM is implicit during evaluation (MLM fills [MASK]).
    # If you want an explicit reconstructor writing text files, we’ll add it next.
    print("PM reconstruction happens on-the-fly during evaluation.")


if __name__ == "__main__":
    main()
